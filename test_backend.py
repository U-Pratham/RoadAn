"""
Stage 4: Backend & Database Integration Test Suite
===================================================
Tests FastAPI endpoints with real model inference, SQLite database persistence,
schema validation, error handling, and security checks.
"""

import os
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.database import get_db_connection, init_db


class TestBackendIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        # Initialize client with lifespan
        cls.client = TestClient(app)
        cls.test_images_dir = Path("data/test_images")
        cls.pothole_img_path = cls.test_images_dir / "01.jpg"
        cls.clean_img_path = cls.test_images_dir / "07.jpg"

    # 1. Health Endpoint
    def test_01_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["model"], "YOLO11n-RDD2022")
        self.assertIn("active_provider", data)
        self.assertIn("hardware_target", data)

    # 2. Real Image Upload & Inference (01.jpg - Pothole)
    def test_02_detect_real_image_pothole(self):
        self.assertTrue(self.pothole_img_path.exists(), "Test image 01.jpg missing")

        with open(self.pothole_img_path, "rb") as f:
            files = {"file": ("01.jpg", f, "image/jpeg")}
            data = {
                "confidence_threshold": "0.25",
                "latitude": "12.971598",
                "longitude": "77.594562",
                "road_category": "main_urban"
            }
            response = self.client.post("/api/detect", files=files, data=data)

        self.assertEqual(response.status_code, 201)
        res = response.json()

        # Validate Schema Fields
        self.assertIn("analysis_id", res)
        self.assertEqual(res["original_filename"], "01.jpg")
        self.assertEqual(res["road_category"], "main_urban")
        self.assertAlmostEqual(res["latitude"], 12.971598, places=4)
        self.assertAlmostEqual(res["longitude"], 77.594562, places=4)
        self.assertGreater(res["detection_count"], 0)
        self.assertGreater(res["overall_risk_score"], 0)
        self.assertIn(res["priority_category"], ["Critical", "High", "Moderate", "Low"])
        self.assertIn("detections", res)
        self.assertGreater(len(res["detections"]), 0)

        # Validate Risk Engine Factor Breakdown on First Detection
        first_det = res["detections"][0]
        self.assertEqual(first_det["damage_class"], "pothole")
        self.assertGreaterEqual(first_det["confidence"], 0.25)
        self.assertEqual(len(first_det["bounding_box"]), 4)
        self.assertIsNotNone(first_det["factors"])
        self.assertEqual(first_det["factors"]["severity"], 40.0)
        self.assertEqual(first_det["factors"]["road_importance"], 18.0) # main_urban

        # Save ID for downstream retrieval tests
        TestBackendIntegration.saved_analysis_id = res["analysis_id"]

    # 3. Database Persistence Verification
    def test_03_database_persistence(self):
        analysis_id = getattr(TestBackendIntegration, "saved_analysis_id", None)
        self.assertIsNotNone(analysis_id, "No analysis_id from previous test")

        conn = get_db_connection()
        cur = conn.cursor()

        # Verify Analysis Record
        cur.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
        analysis_row = cur.fetchone()
        self.assertIsNotNone(analysis_row)
        self.assertEqual(analysis_row["original_filename"], "01.jpg")

        # Verify Child Detection Records
        cur.execute("SELECT * FROM detections WHERE analysis_id = ?", (analysis_id,))
        det_rows = cur.fetchall()
        self.assertGreater(len(det_rows), 0)
        for d in det_rows:
            self.assertEqual(d["analysis_id"], analysis_id)
            self.assertEqual(d["damage_class"], "pothole")
            self.assertGreater(d["individual_risk_score"], 0)

        conn.close()

    # 4. History Listing Endpoint
    def test_04_history_listing(self):
        response = self.client.get("/api/history?limit=10&offset=0")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total", data)
        self.assertIn("items", data)
        self.assertGreater(data["total"], 0)
        self.assertGreater(len(data["items"]), 0)

        # Check item structure
        first_item = data["items"][0]
        self.assertIn("id", first_item)
        self.assertIn("overall_risk_score", first_item)
        self.assertIn("priority_category", first_item)
        self.assertIn("detection_count", first_item)

    # 5. History Detail Endpoint
    def test_05_history_detail(self):
        analysis_id = getattr(TestBackendIntegration, "saved_analysis_id", 1)
        response = self.client.get(f"/api/history/{analysis_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["id"], analysis_id)
        self.assertIn("detections", data)
        self.assertGreater(len(data["detections"]), 0)

        # 404 on non-existent record
        resp_404 = self.client.get("/api/history/999999")
        self.assertEqual(resp_404.status_code, 404)

    # 6. Statistics Endpoint
    def test_06_statistics_endpoint(self):
        response = self.client.get("/api/statistics")
        self.assertEqual(response.status_code, 200)
        stats = response.json()

        self.assertGreaterEqual(stats["total_analyses_count"], 1)
        self.assertGreaterEqual(stats["total_detections_count"], 1)
        self.assertGreater(stats["average_risk_score"], 0.0)
        self.assertGreater(stats["average_inference_latency_ms"], 0.0)
        self.assertIn("pothole", stats["detections_by_class"])
        self.assertIn("main_urban", stats["analyses_by_road_category"])

    # 7. Image Retrieval Endpoint
    def test_07_image_retrieval(self):
        analysis_id = getattr(TestBackendIntegration, "saved_analysis_id", 1)
        # Annotated image
        resp_annotated = self.client.get(f"/api/images/{analysis_id}?type=annotated")
        self.assertEqual(resp_annotated.status_code, 200)
        self.assertEqual(resp_annotated.headers["content-type"], "image/jpeg")
        self.assertGreater(len(resp_annotated.content), 1000)

        # Original image
        resp_orig = self.client.get(f"/api/images/{analysis_id}?type=original")
        self.assertEqual(resp_orig.status_code, 200)
        self.assertGreater(len(resp_orig.content), 1000)

        # 404 on invalid ID
        resp_404 = self.client.get("/api/images/999999")
        self.assertEqual(resp_404.status_code, 404)

    # 8. Missing GPS Handling
    def test_08_missing_gps_handling(self):
        with open(self.pothole_img_path, "rb") as f:
            files = {"file": ("01_no_gps.jpg", f, "image/jpeg")}
            data = {"confidence_threshold": "0.25"} # No lat/lon provided
            response = self.client.post("/api/detect", files=files, data=data)

        self.assertEqual(response.status_code, 201)
        res = response.json()
        self.assertIsNone(res["latitude"])
        self.assertIsNone(res["longitude"])
        self.assertEqual(res["road_category"], "unknown")
        # Factors must reflect neutral unknown road default (15 pts)
        first_det = res["detections"][0]
        self.assertEqual(first_det["factors"]["road_importance"], 15.0)

    # 9. Clean Road (Empty Detections) Handling
    def test_09_clean_road_empty_detections(self):
        self.assertTrue(self.clean_img_path.exists(), "Test image 07.jpg missing")

        with open(self.clean_img_path, "rb") as f:
            files = {"file": ("07.jpg", f, "image/jpeg")}
            data = {"confidence_threshold": "0.25"}
            response = self.client.post("/api/detect", files=files, data=data)

        self.assertEqual(response.status_code, 201)
        res = response.json()
        self.assertEqual(res["detection_count"], 0)
        self.assertEqual(res["overall_risk_score"], 0)
        self.assertEqual(res["priority_category"], "Low")
        self.assertEqual(len(res["detections"]), 0)
        self.assertIn("No road hazards detected", res["summary_explanation"])

    # 10. Security & Validation Checks
    def test_10_invalid_and_corrupt_files(self):
        # A. Unsupported file format (.txt)
        bad_text = b"This is not an image file"
        response = self.client.post(
            "/api/detect",
            files={"file": ("malicious.txt", bad_text, "text/plain")}
        )
        self.assertEqual(response.status_code, 415)

        # B. Invalid GPS Coordinates (latitude > 90)
        with open(self.pothole_img_path, "rb") as f:
            files = {"file": ("01.jpg", f, "image/jpeg")}
            data = {"latitude": "125.0"} # Out of bounds
            response = self.client.post("/api/detect", files=files, data=data)
        self.assertEqual(response.status_code, 422)

        # C. Empty file payload (0 bytes)
        response_empty = self.client.post(
            "/api/detect",
            files={"file": ("empty.jpg", b"", "image/jpeg")}
        )
        self.assertEqual(response_empty.status_code, 400)

    # 11. Demo Samples Endpoint and Mounted Static Files
    def test_11_demo_samples_and_static_mounting(self):
        response = self.client.get("/api/demo-samples")
        self.assertEqual(response.status_code, 200)
        samples = response.json()
        self.assertIsInstance(samples, list)
        self.assertGreaterEqual(len(samples), 5)

        # Verify each demo sample has required fields and its public URL resolves
        for s in samples:
            self.assertIn("id", s)
            self.assertIn("title", s)
            self.assertIn("filename", s)
            self.assertIn("url", s)
            self.assertIn("badge", s)
            # Test public static image retrieval
            img_res = self.client.get(s["url"])
            self.assertEqual(img_res.status_code, 200, f"Demo image {s['url']} failed to load")
            self.assertGreater(len(img_res.content), 0)

    # 12. Applied Confidence Threshold Consistency
    def test_12_applied_confidence_threshold_reporting(self):
        with open(self.pothole_img_path, "rb") as f:
            files = {"file": ("01.jpg", f, "image/jpeg")}
            data = {"confidence_threshold": "0.15"}
            response = self.client.post("/api/detect", files=files, data=data)

        self.assertEqual(response.status_code, 201)
        res = response.json()
        self.assertIn("applied_confidence_threshold", res)
        self.assertAlmostEqual(res["applied_confidence_threshold"], 0.15, places=2)

    # 13. Missing Physical Image Graceful SVG Fallback
    def test_13_missing_physical_image_graceful_svg_fallback(self):
        # Create a record pointing to non-existent files
        from app.database import save_analysis
        ghost_id = save_analysis(
            original_filename="ghost_archived.jpg",
            image_path="nonexistent_folder/ghost.jpg",
            annotated_path="nonexistent_folder/annotated_ghost.jpg",
            latitude=None,
            longitude=None,
            road_category="unknown",
            overall_risk_score=50,
            priority_category="Moderate",
            inference_ms=10.0,
            total_ms=15.0,
            summary_explanation="Archived test record without physical file on disk",
            detections=[]
        )

        resp = self.client.get(f"/api/images/{ghost_id}?type=annotated")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("image/svg+xml", resp.headers["content-type"])
        self.assertIn(b"<svg", resp.content)
        self.assertIn(b"Inspection Audit", resp.content)


if __name__ == "__main__":
    unittest.main()

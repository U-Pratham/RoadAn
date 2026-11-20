"""
RoadSight AI - Stage 6 Comprehensive Final Validation Suite
Executes end-to-end smoke tests, parametric control verification,
performance benchmarking, and failure/edge-case audits.
"""

import os
import sys
import time
import json
import sqlite3
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from app.main import app
from app.database import get_db_connection, init_db
from app.config import DB_PATH, DEFAULT_MODEL_PATH
from app.services.inference_service import InferenceService

class TestStage6FinalValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)
        cls.test_images_dir = Path("data/test_images")
        cls.pothole_img = cls.test_images_dir / "01.jpg"
        cls.clean_img = cls.test_images_dir / "07.jpg"
        cls.multi_defect_img = cls.test_images_dir / "02.jpg"

    # =========================================================================
    # TASK 1 & TASK 2: FULL SMOKE TEST & PARAMETRIC CONTROLS
    # =========================================================================

    def test_01_health_and_hardware_discovery(self):
        """Verify API health and execution provider (CPU or GPU)."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["model"], "YOLO11n-RDD2022")
        self.assertIn(data["active_provider"], ["CPUExecutionProvider", "DmlExecutionProvider", "CUDAExecutionProvider"])
        self.assertIn("hardware_target", data)
        print(f"\n[Hardware] Active Provider: {data['active_provider']} on {data['hardware_target']}")

    def test_02_confidence_threshold_impact(self):
        """Verify that confidence slider changes actual inference detections."""
        self.assertTrue(self.pothole_img.exists())

        # High threshold (0.50): filters out lower confidence detections
        with open(self.pothole_img, "rb") as f:
            r_high = self.client.post("/api/detect", files={"file": ("01.jpg", f, "image/jpeg")}, data={"confidence_threshold": "0.50"})
        self.assertEqual(r_high.status_code, 201)
        count_high = r_high.json()["detection_count"]

        # Moderate threshold (0.25): admits lower confidence detections
        with open(self.pothole_img, "rb") as f:
            r_mod = self.client.post("/api/detect", files={"file": ("01.jpg", f, "image/jpeg")}, data={"confidence_threshold": "0.25"})
        self.assertEqual(r_mod.status_code, 201)
        count_mod = r_mod.json()["detection_count"]

        self.assertGreaterEqual(count_mod, count_high, "Lower threshold should yield at least as many or more detections")
        print(f"[Parametric] Confidence 0.50 detections: {count_high} | Confidence 0.25 detections: {count_mod}")

    def test_03_road_category_impact_on_risk_scoring(self):
        """Verify that road category selection deterministically modulates risk score."""
        with open(self.pothole_img, "rb") as f:
            r_highway = self.client.post("/api/detect", files={"file": ("01.jpg", f, "image/jpeg")}, data={"road_category": "highway", "confidence_threshold": "0.25"})
        score_highway = r_highway.json()["overall_risk_score"]

        with open(self.pothole_img, "rb") as f:
            r_internal = self.client.post("/api/detect", files={"file": ("01.jpg", f, "image/jpeg")}, data={"road_category": "internal", "confidence_threshold": "0.25"})
        score_internal = r_internal.json()["overall_risk_score"]

        # Highway (25 pts) vs Internal (6 pts) -> Delta of ~19 pts on individual factors
        self.assertGreater(score_highway, score_internal)
        print(f"[Parametric] Risk Score on Highway: {score_highway} vs Internal: {score_internal} (Delta: {score_highway - score_internal} pts)")

    def test_04_gps_submission_and_retrieval(self):
        """Verify valid GPS coordinates are persisted and retrieved in history and detail."""
        lat = 12.971598
        lon = 77.594562
        with open(self.pothole_img, "rb") as f:
            res = self.client.post("/api/detect", files={"file": ("gps_test.jpg", f, "image/jpeg")}, data={"latitude": str(lat), "longitude": str(lon), "road_category": "main_urban"})
        self.assertEqual(res.status_code, 201)
        analysis_id = res.json()["analysis_id"]

        # History retrieval
        r_hist = self.client.get(f"/api/history/{analysis_id}")
        self.assertEqual(r_hist.status_code, 200)
        data = r_hist.json()
        self.assertAlmostEqual(data["latitude"], lat, places=5)
        self.assertAlmostEqual(data["longitude"], lon, places=5)
        print(f"[Geospatial] Stored GPS: ({data['latitude']}, {data['longitude']}) for Analysis #{analysis_id}")

    def test_05_missing_gps_handling_unlocated(self):
        """Verify images without GPS have null coordinates and neutral road importance factor."""
        with open(self.pothole_img, "rb") as f:
            res = self.client.post("/api/detect", files={"file": ("no_gps.jpg", f, "image/jpeg")}, data={"road_category": "unknown"})
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertIsNone(data["latitude"])
        self.assertIsNone(data["longitude"])
        self.assertEqual(data["road_category"], "unknown")
        # Factor D must be neutral default: 15.0 pts
        first_det = data["detections"][0]
        self.assertEqual(first_det["factors"]["road_importance"], 15.0)
        print(f"[Geospatial] Missing GPS correctly identified as unlocated with neutral road weight (15.0 pts)")

    def test_06_image_streaming_endpoints(self):
        """Verify binary streaming of both annotated and original images."""
        with open(self.pothole_img, "rb") as f:
            r_det = self.client.post("/api/detect", files={"file": ("img_stream.jpg", f, "image/jpeg")})
        aid = r_det.json()["analysis_id"]

        r_ann = self.client.get(f"/api/images/{aid}?type=annotated")
        self.assertEqual(r_ann.status_code, 200)
        self.assertEqual(r_ann.headers["content-type"], "image/jpeg")
        self.assertGreater(len(r_ann.content), 5000)

        r_orig = self.client.get(f"/api/images/{aid}?type=original")
        self.assertEqual(r_orig.status_code, 200)
        self.assertEqual(r_orig.headers["content-type"], "image/jpeg")
        self.assertGreater(len(r_orig.content), 5000)
        print(f"[Binary Stream] Annotated ({len(r_ann.content)} bytes) and Original ({len(r_orig.content)} bytes) verified")

    # =========================================================================
    # TASK 3: PERFORMANCE BENCHMARKING
    # =========================================================================

    def test_07_latency_breakdown_profiling(self):
        """Measure cold vs warm inference latency, preprocessing, and full HTTP turnaround."""
        service = InferenceService.get_instance()

        # Warm runs
        warm_times = []
        for _ in range(5):
            res = service.predict(self.pothole_img, conf_thresh=0.25)
            warm_times.append(res["inference_ms"])

        avg_warm_infer = sum(warm_times) / len(warm_times)
        min_warm_infer = min(warm_times)

        # Full HTTP roundtrip
        t0 = time.perf_counter()
        with open(self.pothole_img, "rb") as f:
            r_http = self.client.post("/api/detect", files={"file": ("profile.jpg", f, "image/jpeg")})
        http_latency_ms = (time.perf_counter() - t0) * 1000

        self.assertEqual(r_http.status_code, 201)
        resp_data = r_http.json()

        print("\n--- Latency Benchmark Profile ---")
        print(f"Warm ONNX GPU Inference (Avg 5 runs): {avg_warm_infer:.2f} ms (Fastest: {min_warm_infer:.2f} ms)")
        print(f"Backend Server Processing (Reported):  {resp_data['total_processing_ms']:.2f} ms")
        print(f"Full HTTP Ingestion-to-Response:     {http_latency_ms:.2f} ms")
        print("---------------------------------\n")

    # =========================================================================
    # TASK 4: FAILURE & RESILIENCE AUDIT
    # =========================================================================

    def test_08_invalid_and_corrupt_payloads(self):
        """Test rejection of non-image MIME types, out-of-bound coordinates, and empty files."""
        # 1. Non-image format
        r_text = self.client.post("/api/detect", files={"file": ("bad.txt", b"plain text", "text/plain")})
        self.assertEqual(r_text.status_code, 415)

        # 2. Out-of-bounds Latitude (> 90)
        with open(self.pothole_img, "rb") as f:
            r_lat = self.client.post("/api/detect", files={"file": ("01.jpg", f, "image/jpeg")}, data={"latitude": "95.5"})
        self.assertEqual(r_lat.status_code, 422)

        # 3. Out-of-bounds Longitude (> 180)
        with open(self.pothole_img, "rb") as f:
            r_lon = self.client.post("/api/detect", files={"file": ("01.jpg", f, "image/jpeg")}, data={"longitude": "-185.0"})
        self.assertEqual(r_lon.status_code, 422)

        # 4. Zero-byte payload
        r_empty = self.client.post("/api/detect", files={"file": ("empty.jpg", b"", "image/jpeg")})
        self.assertEqual(r_empty.status_code, 400)
        print("[Resilience] Rejections for 415 (MIME), 422 (GPS bounds), and 400 (Empty) verified")

    def test_09_clean_road_zero_defect_resilience(self):
        """Test clean road image produces 0 detections, 0 score, and 'Low' priority."""
        self.assertTrue(self.clean_img.exists())
        with open(self.clean_img, "rb") as f:
            res = self.client.post("/api/detect", files={"file": ("07.jpg", f, "image/jpeg")}, data={"confidence_threshold": "0.25"})
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["detection_count"], 0)
        self.assertEqual(data["overall_risk_score"], 0)
        self.assertEqual(data["priority_category"], "Low")
        self.assertIn("No road hazards detected", data["summary_explanation"])
        print(f"[Resilience] Clean road image (07.jpg): {data['detection_count']} detections, Score: {data['overall_risk_score']}, Priority: {data['priority_category']}")

    def test_10_multiple_detections_and_factor_bounds(self):
        """Test image with multiple defects computes individual scores and bounding boxes."""
        self.assertTrue(self.pothole_img.exists())
        with open(self.pothole_img, "rb") as f:
            res = self.client.post("/api/detect", files={"file": ("01.jpg", f, "image/jpeg")}, data={"confidence_threshold": "0.25"})
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertGreaterEqual(len(data["detections"]), 2)
        for d in data["detections"]:
            self.assertGreaterEqual(d["individual_risk_score"], 0)
            self.assertLessEqual(d["individual_risk_score"], 100)
            factors = d["factors"]
            self.assertLessEqual(factors["severity"], 40.0)
            self.assertLessEqual(factors["area"], 20.0)
            self.assertLessEqual(factors["confidence"], 15.0)
            self.assertLessEqual(factors["road_importance"], 25.0)
        print(f"[Resilience] Multi-defect verification: {len(data['detections'])} detections all strictly bounded within [0, 100]")

    def test_11_database_persistence_across_reconnect(self):
        """Verify SQLite data persistence across connection lifecycles."""
        with open(self.pothole_img, "rb") as f:
            r = self.client.post("/api/detect", files={"file": ("persist_test.jpg", f, "image/jpeg")})
        aid = r.json()["analysis_id"]

        # Simulate fresh DB connection
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM analyses WHERE id = ?", (aid,))
        analysis_row = cur.fetchone()
        self.assertIsNotNone(analysis_row)
        self.assertEqual(analysis_row["original_filename"], "persist_test.jpg")

        cur.execute("SELECT COUNT(*) FROM detections WHERE analysis_id = ?", (aid,))
        child_count = cur.fetchone()[0]
        self.assertGreater(child_count, 0)
        conn.close()
        print(f"[Persistence] Analysis #{aid} and {child_count} child detections safely verified in SQLite disk file")

    def test_12_statistics_live_consistency(self):
        """Verify /api/statistics accurately aggregates database records."""
        res = self.client.get("/api/statistics")
        self.assertEqual(res.status_code, 200)
        stats = res.json()
        self.assertGreater(stats["total_analyses_count"], 0)
        self.assertGreater(stats["total_detections_count"], 0)
        self.assertGreater(stats["average_risk_score"], 0)
        self.assertIn("pothole", stats["detections_by_class"])
        print(f"[Statistics] Aggregates: Total Analyses={stats['total_analyses_count']}, Total Hazards={stats['total_detections_count']}, Avg Risk={stats['average_risk_score']}")

if __name__ == "__main__":
    unittest.main()

"""
Stage 5: Frontend & Full Application Integration Test
Verifies that static assets are correctly served, API routes connect cleanly,
and real end-to-end inspection flows execute as expected by the frontend.
"""

import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
from app.database import init_db

class TestFrontendIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)
        cls.test_img_path = Path("data/test_images/01.jpg")

    def test_01_index_html_served(self):
        """Verify root / serves index.html with RoadSight UI markup."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers.get("content-type", ""))
        self.assertIn("ROADSIGHT", res.text)
        self.assertIn("section-detect", res.text)
        self.assertIn("section-map", res.text)
        self.assertIn("section-history", res.text)

    def test_02_static_css_and_js_served(self):
        """Verify static CSS and JS assets are accessible."""
        res_css = self.client.get("/static/style.css")
        self.assertEqual(res_css.status_code, 200)
        self.assertIn("--bg-base", res_css.text)

        res_js = self.client.get("/static/app.js")
        self.assertEqual(res_js.status_code, 200)
        self.assertIn("RoadSight AI", res_js.text)
        self.assertIn("PRIORITY_CONFIG", res_js.text)

    def test_03_frontend_api_flow(self):
        """Simulate browser frontend: upload image, fetch stats, fetch history."""
        self.assertTrue(self.test_img_path.exists())

        # 1. Ingest image as frontend does
        with open(self.test_img_path, "rb") as f:
            files = {"file": ("frontend_test.jpg", f, "image/jpeg")}
            data = {
                "confidence_threshold": "0.25",
                "latitude": "12.9716",
                "longitude": "77.5946",
                "road_category": "highway"
            }
            res_detect = self.client.post("/api/detect", files=files, data=data)

        self.assertEqual(res_detect.status_code, 201)
        det_json = res_detect.json()
        analysis_id = det_json["analysis_id"]
        self.assertGreater(det_json["overall_risk_score"], 0)
        self.assertIn(det_json["priority_category"], ["Critical", "High", "Moderate", "Low"])
        self.assertIn("detections", det_json)

        # 2. Fetch statistics as frontend does
        res_stats = self.client.get("/api/statistics")
        self.assertEqual(res_stats.status_code, 200)
        stats_json = res_stats.json()
        self.assertGreaterEqual(stats_json["total_analyses_count"], 1)

        # 3. Fetch history as frontend does
        res_hist = self.client.get("/api/history?limit=100&offset=0")
        self.assertEqual(res_hist.status_code, 200)
        hist_json = res_hist.json()
        self.assertGreater(hist_json["total"], 0)
        
        # Verify the analysis is present
        found = any(item["id"] == analysis_id for item in hist_json["items"])
        self.assertTrue(found)

        # 4. Fetch detail modal data as frontend does
        res_detail = self.client.get(f"/api/history/{analysis_id}")
        self.assertEqual(res_detail.status_code, 200)
        detail_json = res_detail.json()
        self.assertEqual(detail_json["id"], analysis_id)

        # 5. Fetch image streaming as frontend does
        res_img = self.client.get(f"/api/images/{analysis_id}?type=annotated")
        self.assertEqual(res_img.status_code, 200)
        self.assertEqual(res_img.headers["content-type"], "image/jpeg")

if __name__ == "__main__":
    unittest.main()

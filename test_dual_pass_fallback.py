"""
RoadSight AI - Dual-Pass Fallback Test Suite
============================================
Validates Automatic Dual-Pass Fallback (Option 1):
  1. Standard inference pass executes first without preprocessing modifications.
  2. Enhanced gamma=1.5 pass runs ONLY when standard pass returns 0 detections.
  3. Validates wet-road puddle pothole detection and proper labeling.
  4. Confirms dry roads with detections do not trigger Pass 2.
  5. Confirms clean undamaged roads produce 0 false positives across both passes.
  6. Verifies bounding box coordinates and area calculations.
  7. Audits pass 1 and pass 2 telemetry.
"""

import unittest
from pathlib import Path
from app.services.inference_service import InferenceService
from app.services.risk_service import RiskService
from risk_engine import assess_image_risk

class TestDualPassFallback(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.infer_service = InferenceService.get_instance()
        cls.wet_pothole_path = Path("data/uploads/ebf5a043553e45be892326dc262021e5_OIP1.webp")
        cls.dry_road_path = Path("data/test_images/01.jpg")
        cls.multi_pothole_path = Path("data/test_images/04.jpg")
        cls.clean_road_path = Path("data/test_images/07.jpg")
        cls.clean_test1_path = Path("data/test_images/test1.jpg")
        cls.clean_test4_path = Path("data/test_images/test4.jpg")
        cls.clean_test5_path = Path("data/test_images/test5.jpg")

    def test_01_wet_road_puddle_fallback_activation(self):
        """Verify wet road puddle pothole misses Pass 1, activates Pass 2, and detects pothole."""
        if not self.wet_pothole_path.exists():
            self.skipTest(f"Missing {self.wet_pothole_path}")

        res = self.infer_service.predict(self.wet_pothole_path, conf_thresh=0.20)
        self.assertTrue(res["dual_pass_executed"], "Dual-pass should execute when Pass 1 yields 0")
        self.assertEqual(res["pass1_count"], 0, "Pass 1 must produce 0 detections on raw water glare")
        self.assertGreater(res["pass2_count"], 0, "Pass 2 must produce enhanced detections")
        self.assertEqual(res["inference_pass_used"], "enhanced_wet_road")

        # Verify detection properties
        det = res["detections"][0]
        self.assertEqual(det["class_name"], "pothole")
        self.assertGreaterEqual(det["score"], 0.20)
        self.assertEqual(det["inference_pass"], "enhanced_wet_road")
        self.assertEqual(det["enhancement_note"], "Detected via Wet-Road / Puddle Contrast Boost")

        # Verify bounding box bounds & area
        box = det["box"]
        self.assertGreaterEqual(box[0], 0.0)
        self.assertGreaterEqual(box[1], 0.0)
        self.assertLessEqual(box[2], float(res["orig_w"]))
        self.assertLessEqual(box[3], float(res["orig_h"]))
        w = box[2] - box[0]
        h = box[3] - box[1]
        self.assertGreater(w, 10.0)
        self.assertGreater(h, 10.0)

    def test_02_dry_road_never_triggers_pass2(self):
        """Verify normal dry roads with detections complete on Pass 1 and do not trigger Pass 2."""
        res = self.infer_service.predict(self.dry_road_path, conf_thresh=0.20)
        self.assertFalse(res["dual_pass_executed"], "Pass 2 must NOT run when Pass 1 finds detections")
        self.assertEqual(res["inference_pass_used"], "standard")
        self.assertGreater(len(res["detections"]), 0)
        for d in res["detections"]:
            self.assertEqual(d["inference_pass"], "standard")
            self.assertIsNone(d["enhancement_note"])

    def test_03_multi_pothole_never_triggers_pass2(self):
        """Verify multi-pothole dry road completes on Pass 1."""
        res = self.infer_service.predict(self.multi_pothole_path, conf_thresh=0.20)
        self.assertFalse(res["dual_pass_executed"])
        self.assertEqual(res["inference_pass_used"], "standard")
        self.assertEqual(len(res["detections"]), 5)

    def test_04_clean_road_zero_false_positives(self):
        """Verify clean road runs Pass 1, runs Pass 2, and produces 0 false positives across both."""
        res = self.infer_service.predict(self.clean_road_path, conf_thresh=0.20)
        self.assertTrue(res["dual_pass_executed"])
        self.assertEqual(res["pass1_count"], 0)
        self.assertEqual(res["pass2_count"], 0)
        self.assertEqual(len(res["detections"]), 0)
        self.assertEqual(res["inference_pass_used"], "standard")

    def test_05_additional_clean_roads_zero_false_positives(self):
        """Verify additional smooth test roads produce 0 detections across both passes."""
        for p in [self.clean_test1_path, self.clean_test4_path, self.clean_test5_path]:
            if p.exists():
                res = self.infer_service.predict(p, conf_thresh=0.20)
                self.assertEqual(len(res["detections"]), 0, f"False positive detected on clean road {p.name}")

    def test_06_risk_engine_integration_and_audit_labeling(self):
        """Verify risk engine incorporates dual-pass metadata and does not alter scoring formulas."""
        if not self.wet_pothole_path.exists():
            self.skipTest(f"Missing {self.wet_pothole_path}")

        res = self.infer_service.predict(self.wet_pothole_path, conf_thresh=0.20)
        summary = RiskService.evaluate(
            detections=res["detections"],
            image_width=res["orig_w"],
            image_height=res["orig_h"],
            road_category="internal",
            dual_pass_executed=res["dual_pass_executed"],
            inference_pass_used=res["inference_pass_used"]
        )

        self.assertGreater(summary.image_risk_score, 60)
        self.assertIn("Wet-Road / Puddle Contrast Boost Active", summary.summary_explanation)
        self.assertEqual(summary.individual_assessments[0].inference_pass, "enhanced_wet_road")
        self.assertIn("Detected via Wet-Road / Puddle Contrast Boost", summary.individual_assessments[0].explanation)

    def test_07_zero_detection_uncertainty_disclaimer(self):
        """Verify dual-pass zero detections outputs uncertainty warning rather than declaring road safe."""
        summary = assess_image_risk([], 640, 640, road_category="main_urban", dual_pass_executed=True)
        self.assertEqual(summary.image_risk_score, 0)
        self.assertIn("No road hazards detected above the confidence threshold", summary.summary_explanation)
        self.assertIn("Note: Visual AI scan does not guarantee structural integrity", summary.summary_explanation)
        self.assertNotIn("acceptable condition", summary.summary_explanation)

if __name__ == "__main__":
    unittest.main()

"""
Unit Tests for RoadSight AI Stage 3 Risk Intelligence Engine
============================================================
Tests cover:
  1. Every damage class (pothole, alligator_crack, transverse_crack, longitudinal_crack).
  2. Every road importance category (highway, main_urban, residential, internal, unknown).
  3. Low and high confidence scaling.
  4. Small and large bounding box area scaling.
  5. Missing, invalid, or unknown road categories.
  6. Multiple detections and non-linear risk aggregation (overlap discounting).
  7. Invalid and edge-case bounding-box coordinates.
  8. Score boundaries and strict [0, 100] clamping.
  9. Determinism of repeated calculations.
"""

import unittest
from risk_engine import (
    assess_detection_risk,
    assess_image_risk,
    normalize_road_category,
    get_priority_category,
    RoadCategory,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_MODERATE,
    PRIORITY_LOW,
    DAMAGE_SEVERITY_WEIGHTS,
    ROAD_IMPORTANCE_WEIGHTS
)


class TestRiskEngine(unittest.TestCase):

    def setUp(self):
        self.img_w = 640
        self.img_h = 640

    # 1. Test Every Damage Class
    def test_every_damage_class(self):
        expected_classes = ["pothole", "alligator_crack", "transverse_crack", "longitudinal_crack"]
        for d_class in expected_classes:
            res = assess_detection_risk(
                damage_class=d_class,
                confidence=0.5,
                bounding_box=[100, 100, 200, 200],
                image_width=self.img_w,
                image_height=self.img_h,
                road_category="main_urban"
            )
            self.assertEqual(res.damage_class, d_class)
            self.assertEqual(res.factors.severity, DAMAGE_SEVERITY_WEIGHTS[d_class])
            self.assertGreaterEqual(res.risk_score, 0)
            self.assertLessEqual(res.risk_score, 100)

        # Hierarchy assertion: Pothole > Alligator > Transverse > Longitudinal under identical conditions
        res_pothole = assess_detection_risk("pothole", 0.5, [100, 100, 200, 200], self.img_w, self.img_h, "main_urban")
        res_alligator = assess_detection_risk("alligator_crack", 0.5, [100, 100, 200, 200], self.img_w, self.img_h, "main_urban")
        res_transverse = assess_detection_risk("transverse_crack", 0.5, [100, 100, 200, 200], self.img_w, self.img_h, "main_urban")
        res_longitudinal = assess_detection_risk("longitudinal_crack", 0.5, [100, 100, 200, 200], self.img_w, self.img_h, "main_urban")

        self.assertGreater(res_pothole.risk_score, res_alligator.risk_score)
        self.assertGreater(res_alligator.risk_score, res_transverse.risk_score)
        self.assertGreater(res_transverse.risk_score, res_longitudinal.risk_score)

    # 2. Test Every Road Importance Category
    def test_every_road_category(self):
        categories = ["highway", "main_urban", "residential", "internal", "unknown"]
        for cat in categories:
            res = assess_detection_risk(
                damage_class="pothole",
                confidence=0.5,
                bounding_box=[100, 100, 200, 200],
                image_width=self.img_w,
                image_height=self.img_h,
                road_category=cat
            )
            enum_cat = normalize_road_category(cat)
            self.assertEqual(res.road_category, enum_cat.value)
            self.assertEqual(res.factors.road_importance, ROAD_IMPORTANCE_WEIGHTS[enum_cat])

        # Highway > Main Urban > Residential > Internal
        res_hw = assess_detection_risk("pothole", 0.5, [100, 100, 200, 200], self.img_w, self.img_h, "highway")
        res_mu = assess_detection_risk("pothole", 0.5, [100, 100, 200, 200], self.img_w, self.img_h, "main_urban")
        res_res = assess_detection_risk("pothole", 0.5, [100, 100, 200, 200], self.img_w, self.img_h, "residential")
        res_int = assess_detection_risk("pothole", 0.5, [100, 100, 200, 200], self.img_w, self.img_h, "internal")

        self.assertGreater(res_hw.risk_score, res_mu.risk_score)
        self.assertGreater(res_mu.risk_score, res_res.risk_score)
        self.assertGreater(res_res.risk_score, res_int.risk_score)

    # 3. Test Low and High Confidence
    def test_low_and_high_confidence(self):
        res_low = assess_detection_risk("pothole", 0.15, [100, 100, 200, 200], self.img_w, self.img_h, "main_urban")
        res_high = assess_detection_risk("pothole", 0.95, [100, 100, 200, 200], self.img_w, self.img_h, "main_urban")

        self.assertAlmostEqual(res_low.factors.confidence, 15.0 * 0.15, places=2)
        self.assertAlmostEqual(res_high.factors.confidence, 15.0 * 0.95, places=2)
        self.assertGreater(res_high.risk_score, res_low.risk_score)

    # 4. Test Small and Large Bounding Boxes
    def test_small_and_large_boxes(self):
        # Tiny box: 10x10 px (100 px^2 out of 409600 = 0.024% of image)
        res_tiny = assess_detection_risk("pothole", 0.5, [100, 100, 110, 110], self.img_w, self.img_h, "main_urban")
        # Massive box: 300x300 px (90000 px^2 = 21.9% of image, exceeds 5% saturation)
        res_huge = assess_detection_risk("pothole", 0.5, [100, 100, 400, 400], self.img_w, self.img_h, "main_urban")

        self.assertLess(res_tiny.factors.area, 1.0)
        self.assertEqual(res_huge.factors.area, 20.0)  # Capped at max 20 points
        self.assertGreater(res_huge.risk_score, res_tiny.risk_score)

    # 5. Test Missing or Unknown Road Category
    def test_missing_or_unknown_road_category(self):
        res_none = assess_detection_risk("pothole", 0.5, [100, 100, 200, 200], self.img_w, self.img_h, None)
        res_empty = assess_detection_risk("pothole", 0.5, [100, 100, 200, 200], self.img_w, self.img_h, "")
        res_rand = assess_detection_risk("pothole", 0.5, [100, 100, 200, 200], self.img_w, self.img_h, "random_road_123")

        self.assertEqual(res_none.road_category, "unknown")
        self.assertEqual(res_empty.road_category, "unknown")
        self.assertEqual(res_rand.road_category, "unknown")
        self.assertEqual(res_none.factors.road_importance, 15.0)  # Neutral default

    # 6. Test Multiple Detections and Non-Linear Aggregation
    def test_multiple_detections(self):
        dets = [
            {"class_name": "pothole", "score": 0.85, "box": [100, 100, 250, 250]},  # Severe pothole
            {"class_name": "alligator_crack", "score": 0.60, "box": [400, 400, 500, 500]}, # Disjoint secondary
            {"class_name": "longitudinal_crack", "score": 0.40, "box": [50, 50, 80, 200]}  # Third small
        ]

        summary = assess_image_risk(dets, self.img_w, self.img_h, road_category="main_urban")
        self.assertEqual(summary.total_detections, 3)
        self.assertIsNotNone(summary.highest_hazard)
        self.assertEqual(summary.highest_hazard.damage_class, "pothole")

        # The image-level risk must be greater than or equal to the highest single hazard,
        # but strictly less than a naive linear sum
        highest_score = summary.highest_hazard.risk_score
        linear_sum = sum(a.risk_score for a in summary.individual_assessments)
        self.assertGreaterEqual(summary.image_risk_score, highest_score)
        self.assertLess(summary.image_risk_score, linear_sum)

        # Overlapping boxes (IoU > 0.5 describing same damage) must NOT significantly inflate risk
        overlapping_dets = [
            {"class_name": "pothole", "score": 0.85, "box": [100, 100, 250, 250]},
            {"class_name": "pothole", "score": 0.80, "box": [105, 105, 255, 255]}  # 90%+ overlapping duplicate
        ]
        overlap_summary = assess_image_risk(overlapping_dets, self.img_w, self.img_h, road_category="main_urban")
        # Overlapping box should produce almost zero increase over base score
        self.assertLessEqual(overlap_summary.image_risk_score, overlap_summary.highest_hazard.risk_score + 2)

    # 7. Test Empty Detections
    def test_empty_detections(self):
        summary = assess_image_risk([], self.img_w, self.img_h, road_category="highway")
        self.assertEqual(summary.image_risk_score, 0)
        self.assertEqual(summary.image_priority, PRIORITY_LOW)
        self.assertIsNone(summary.highest_hazard)

    # 8. Test Invalid Coordinates and Negative Values
    def test_invalid_coordinates(self):
        # Inverted coordinates [x2, y2, x1, y1]
        res_inv = assess_detection_risk("pothole", 0.5, [250, 250, 100, 100], self.img_w, self.img_h, "main_urban")
        self.assertGreater(res_inv.factors.area, 0.0)

        # Negative / out-of-bound coordinates
        res_neg = assess_detection_risk("pothole", 0.5, [-50, -50, 700, 700], self.img_w, self.img_h, "main_urban")
        self.assertLessEqual(res_neg.factors.area, 20.0)

        # Zero or negative image dimensions
        res_zero_dim = assess_detection_risk("pothole", 0.5, [10, 10, 50, 50], 0, 0, "main_urban")
        self.assertEqual(res_zero_dim.factors.area, 0.0)

    # 9. Test Score Boundaries and Clamping
    def test_score_boundaries(self):
        # Maximum possible input (Massive pothole on highway with 1.0 confidence)
        res_max = assess_detection_risk("pothole", 1.0, [0, 0, 640, 640], self.img_w, self.img_h, "highway")
        self.assertEqual(res_max.risk_score, 100)
        self.assertEqual(res_max.priority, PRIORITY_CRITICAL)

        # Minimum realistic input (Longitudinal crack, 0 confidence, tiny box, internal road)
        res_min = assess_detection_risk("longitudinal_crack", 0.0, [0, 0, 1, 1], self.img_w, self.img_h, "internal")
        self.assertGreaterEqual(res_min.risk_score, 0)
        self.assertLessEqual(res_min.risk_score, 34)
        self.assertEqual(res_min.priority, PRIORITY_LOW)

    # 10. Test Determinism of Repeated Calculations
    def test_deterministic_execution(self):
        res1 = assess_detection_risk("alligator_crack", 0.654, [123.4, 234.5, 345.6, 456.7], 1920, 1080, "main_urban")
        res2 = assess_detection_risk("alligator_crack", 0.654, [123.4, 234.5, 345.6, 456.7], 1920, 1080, "main_urban")

        self.assertEqual(res1.risk_score, res2.risk_score)
        self.assertEqual(res1.priority, res2.priority)
        self.assertEqual(res1.factors.total, res2.factors.total)
        self.assertEqual(res1.explanation, res2.explanation)


if __name__ == "__main__":
    unittest.main()

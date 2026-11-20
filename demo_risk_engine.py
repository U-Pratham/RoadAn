"""
RoadSight AI - Stage 3 Risk Engine Live Demonstration
Evaluates:
  1. The same physical pothole across 3 different road types (Highway vs Main Urban vs Residential vs Internal).
  2. Multi-hazard real detection from 01.jpg (dual potholes).
  3. Multi-hazard real detection from 08.jpg (alligator crack + longitudinal crack).
"""

from risk_engine import assess_detection_risk, assess_image_risk

def run_scenarios():
    print("=" * 80)
    print("SCENARIO 1: SAME DAMAGE DEFECT ACROSS DIFFERENT ROAD CATEGORIES")
    print("Fixed Defect: Pothole, Confidence: 0.80, Box: [200, 300, 350, 420] on 720x720 image")
    print("-" * 80)

    road_types = ["highway", "main_urban", "residential", "internal", "unknown"]
    for r_type in road_types:
        res = assess_detection_risk(
            damage_class="pothole",
            confidence=0.80,
            bounding_box=[200, 300, 350, 420],
            image_width=720,
            image_height=720,
            road_category=r_type
        )
        print(f"Road Category: {r_type.upper():12s} | Risk Score: {res.risk_score:3d}/100 | Priority: {res.priority:8s}")
        print(f"  Factors: Severity={res.factors.severity}/40, Area={res.factors.area}/20, Conf={res.factors.confidence}/15, Road={res.factors.road_importance}/25")
        print(f"  Explanation: {res.explanation}")
        print()

    print("=" * 80)
    print("SCENARIO 2: REAL DETECTION FROM 01.JPG (DUAL POTHOLES)")
    print("-" * 80)
    # Detections from Stage 1 / Stage 2 for 01.jpg
    dets_01 = [
        {"class_name": "pothole", "score": 0.509, "box": [460.7, 614.6, 543.3, 663.0]},
        {"class_name": "pothole", "score": 0.388, "box": [431.5, 679.3, 500.0, 704.0]}
    ]
    summary_01 = assess_image_risk(dets_01, 720, 720, road_category="main_urban")
    print(f"Image: 01.jpg | Overall Risk: {summary_01.image_risk_score}/100 | Priority: {summary_01.image_priority}")
    print(f"Primary Hazard: {summary_01.highest_hazard.damage_class} (Risk: {summary_01.highest_hazard.risk_score})")
    print(f"Summary: {summary_01.summary_explanation}")
    print()

    print("=" * 80)
    print("SCENARIO 3: REAL DETECTION FROM 08.JPG (ALLIGATOR CRACK + LONGITUDINAL CRACK)")
    print("-" * 80)
    # Detections from Stage 1 / Stage 2 for 08.jpg
    dets_08 = [
        {"class_name": "alligator_crack", "score": 0.681, "box": [453.6, 424.3, 584.0, 521.8]},
        {"class_name": "longitudinal_crack", "score": 0.316, "box": [341.6, 522.6, 401.7, 597.4]},
        {"class_name": "alligator_crack", "score": 0.258, "box": [490.4, 420.7, 563.8, 477.5]}
    ]
    summary_08 = assess_image_risk(dets_08, 720, 720, road_category="highway")
    print(f"Image: 08.jpg | Overall Risk: {summary_08.image_risk_score}/100 | Priority: {summary_08.image_priority}")
    print(f"Primary Hazard: {summary_08.highest_hazard.damage_class} (Risk: {summary_08.highest_hazard.risk_score})")
    print(f"Summary: {summary_08.summary_explanation}")
    print("=" * 80)

if __name__ == "__main__":
    run_scenarios()

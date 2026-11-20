"""
RoadSight AI - Risk Intelligence Engine
Calculates Road Hazard Index (0-100) and priority actionable insights.
"""

from typing import List, Dict, Any

WEIGHTS = {
    "pothole": 35.0,             # High structural threat to vehicles and safety
    "alligator_crack": 25.0,     # Deep base-layer fatigue failure
    "transverse_crack": 20.0,    # Perpendicular thermal / joint distress
    "longitudinal_crack": 15.0   # Linear wheel-path fatigue
}

ACTION_RECOMMENDATIONS = {
    "CRITICAL": "Emergency repair needed: Severe road cratering poses imminent risk of tire blowout or vehicular accidents.",
    "HIGH": "Priority resurfacing recommended: Significant structural degradation and water ingress risk.",
    "MODERATE": "Preventative maintenance scheduled: Pavement cracks should be crack-sealed to avoid sub-base erosion.",
    "LOW": "No high-severity hazards detected above threshold: Routine monitoring recommended."
}

def calculate_risk(detections: List[Dict[str, Any]], img_w: int = 640, img_h: int = 640) -> Dict[str, Any]:
    """
    Computes Hazard Severity Index (0-100) based on defect types, count,
    confidence, and relative area impact.
    """
    if not detections:
        return {
            "score": 0,
            "level": "LOW",
            "pci": 100,  # Pavement Condition Index (100 = Perfect)
            "color": "#10b981", # Green
            "action": ACTION_RECOMMENDATIONS["LOW"],
            "defect_counts": {"pothole": 0, "alligator_crack": 0, "longitudinal_crack": 0, "transverse_crack": 0}
        }

    raw_score = 0.0
    defect_counts = {"pothole": 0, "alligator_crack": 0, "longitudinal_crack": 0, "transverse_crack": 0}
    total_area = float(img_w * img_h)

    for det in detections:
        cls_name = det.get("class_name", "pothole")
        conf = float(det.get("score", 0.5))
        box = det.get("box", [0, 0, 0, 0])
        
        defect_counts[cls_name] = defect_counts.get(cls_name, 0) + 1

        # Calculate bounding box area ratio
        bw = max(0.0, box[2] - box[0])
        bh = max(0.0, box[3] - box[1])
        area_ratio = min(1.0, (bw * bh) / (total_area * 0.15)) # 15% of frame is huge

        base_weight = WEIGHTS.get(cls_name, 20.0)
        
        # Severity contribution = base_weight * confidence * (0.7 + 0.3 * area_ratio)
        contribution = base_weight * conf * (0.7 + 0.3 * area_ratio)
        raw_score += contribution

    # Normalize to 0-100 range with soft saturation
    final_score = min(100, int(round(raw_score)))

    # Determine risk level
    if final_score >= 70:
        level = "CRITICAL"
        color = "#ef4444" # Red
    elif final_score >= 45:
        level = "HIGH"
        color = "#f97316" # Orange
    elif final_score >= 20:
        level = "MODERATE"
        color = "#eab308" # Yellow
    else:
        level = "LOW"
        color = "#10b981" # Green

    pci = max(0, 100 - final_score)

    return {
        "score": final_score,
        "level": level,
        "pci": pci,
        "color": color,
        "action": ACTION_RECOMMENDATIONS[level],
        "defect_counts": defect_counts
    }

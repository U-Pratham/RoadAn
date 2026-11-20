"""
RoadSight AI - Stage 3: Explainable Road Risk Intelligence Engine
================================================================
Transforms raw road damage detections into actionable, transparent,
and deterministic risk assessments.

Risk Score Formula:
  Total Risk Score (0-100) = Clamped(
      Factor A: Damage Severity       (Max 40 pts)
    + Factor B: Estimated Damaged Area (Max 20 pts)
    + Factor C: Detection Confidence   (Max 15 pts)
    + Factor D: Road Importance        (Max 25 pts)
  )

Priority Categories:
  - Critical:  80 - 100
  - High:      60 - 79
  - Moderate:  35 - 59
  - Low:        0 - 34

DISCLAIMER:
  These factor weights and scoring rules represent engineering heuristics for
  prioritization and triage. They are NOT certified structural engineering
  measurements or scientifically validated road-safety predictions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple


class RoadCategory(str, Enum):
    HIGHWAY = "highway"
    MAIN_URBAN = "main_urban"
    RESIDENTIAL = "residential"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


# Factor A: Damage Severity (Max 40 points)
# Engineering assumptions: Potholes represent sudden kinetic hazard to tires/suspensions;
# Alligator cracking reflects deep base-course failure; Transverse cracks cause ride roughness;
# Longitudinal cracks reflect early-stage linear wheel-path fatigue.
DAMAGE_SEVERITY_WEIGHTS: Dict[str, float] = {
    "pothole": 40.0,
    "alligator_crack": 30.0,
    "transverse_crack": 22.0,
    "longitudinal_crack": 18.0
}

# Factor D: Road Importance (Max 25 points)
# Reflects traffic volume, transit velocity, and consequence of road disruption.
ROAD_IMPORTANCE_WEIGHTS: Dict[RoadCategory, float] = {
    RoadCategory.HIGHWAY: 25.0,
    RoadCategory.MAIN_URBAN: 18.0,
    RoadCategory.RESIDENTIAL: 12.0,
    RoadCategory.INTERNAL: 6.0,
    RoadCategory.UNKNOWN: 15.0  # Neutral default for unclassified roads
}

# Factor B Saturation Threshold: A single bounding box covering >= 5% of
# the image frame reaches the maximum 20 points for visual area extent.
AREA_RATIO_SATURATION = 0.05

# Priority Classifications
PRIORITY_CRITICAL = "Critical"
PRIORITY_HIGH = "High"
PRIORITY_MODERATE = "Moderate"
PRIORITY_LOW = "Low"


@dataclass
class FactorScores:
    severity: float
    area: float
    confidence: float
    road_importance: float
    total: float


@dataclass
class SingleRiskAssessment:
    damage_class: str
    confidence: float
    bounding_box: List[float]
    road_category: str
    risk_score: int
    priority: str
    factors: FactorScores
    explanation: str


@dataclass
class ImageRiskSummary:
    highest_hazard: Optional[SingleRiskAssessment]
    image_risk_score: int
    image_priority: str
    total_detections: int
    hazard_breakdown: Dict[str, int]
    individual_assessments: List[SingleRiskAssessment]
    summary_explanation: str


def normalize_road_category(category: Optional[str]) -> RoadCategory:
    """Normalize input string to supported RoadCategory enum."""
    if not category:
        return RoadCategory.UNKNOWN

    norm = category.strip().lower().replace(" ", "_").replace("-", "_")
    for member in RoadCategory:
        if norm == member.value:
            return member

    # Common aliases
    if norm in ("arterial", "expressway", "freeway", "highway_major_arterial"):
        return RoadCategory.HIGHWAY
    elif norm in ("city", "urban", "major_road", "main_road"):
        return RoadCategory.MAIN_URBAN
    elif norm in ("local", "neighborhood", "suburb", "residential_road"):
        return RoadCategory.RESIDENTIAL
    elif norm in ("parking", "service", "campus", "internal_low_traffic"):
        return RoadCategory.INTERNAL

    return RoadCategory.UNKNOWN


def get_priority_category(score: float) -> str:
    """Assign standardized priority level based on clamped score."""
    clamped = max(0.0, min(100.0, score))
    if clamped >= 80.0:
        return PRIORITY_CRITICAL
    elif clamped >= 60.0:
        return PRIORITY_HIGH
    elif clamped >= 35.0:
        return PRIORITY_MODERATE
    else:
        return PRIORITY_LOW


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union for coordinate lists [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0.0 else 0.0


def assess_detection_risk(
    damage_class: str,
    confidence: float,
    bounding_box: List[float],
    image_width: int,
    image_height: int,
    road_category: Optional[str] = "unknown"
) -> SingleRiskAssessment:
    """
    Assess risk for a single detected road defect.
    
    Parameters:
      damage_class: 'pothole', 'alligator_crack', 'longitudinal_crack', 'transverse_crack'
      confidence: Detection score between 0.0 and 1.0
      bounding_box: [x1, y1, x2, y2]
      image_width: Width of image in pixels (must be > 0)
      image_height: Height of image in pixels (must be > 0)
      road_category: 'highway', 'main_urban', 'residential', 'internal', or 'unknown'
    """
    # Defensive normalization
    norm_class = damage_class.strip().lower()
    norm_cat = normalize_road_category(road_category)

    # 1. Factor A: Damage Severity (Max 40)
    score_severity = DAMAGE_SEVERITY_WEIGHTS.get(norm_class, 15.0)

    # 2. Factor B: Estimated Damaged Area (Max 20)
    # Validate image dimensions and coordinates
    if image_width <= 0 or image_height <= 0:
        area_ratio = 0.0
    else:
        total_pixels = float(image_width * image_height)
        # Handle coordinate anomalies gracefully
        x1 = min(bounding_box[0], bounding_box[2])
        y1 = min(bounding_box[1], bounding_box[3])
        x2 = max(bounding_box[0], bounding_box[2])
        y2 = max(bounding_box[1], bounding_box[3])

        # Clip coordinates within image frame
        x1 = max(0.0, min(float(image_width), x1))
        x2 = max(0.0, min(float(image_width), x2))
        y1 = max(0.0, min(float(image_height), y1))
        y2 = max(0.0, min(float(image_height), y2))

        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        area_ratio = (bw * bh) / total_pixels

    # Linear scaling up to saturation threshold
    score_area = min(20.0, 20.0 * (area_ratio / AREA_RATIO_SATURATION))

    # 3. Factor C: Detection Confidence (Max 15)
    clamped_conf = max(0.0, min(1.0, float(confidence)))
    score_conf = 15.0 * clamped_conf

    # 4. Factor D: Road Importance (Max 25)
    score_road = ROAD_IMPORTANCE_WEIGHTS.get(norm_cat, 15.0)

    # Total Score Calculation (Deterministic & Clamped)
    raw_total = score_severity + score_area + score_conf + score_road
    final_score = int(round(max(0.0, min(100.0, raw_total))))
    priority = get_priority_category(final_score)

    factors = FactorScores(
        severity=round(score_severity, 2),
        area=round(score_area, 2),
        confidence=round(score_conf, 2),
        road_importance=round(score_road, 2),
        total=round(raw_total, 2)
    )

    # Human-readable explanation
    explanation_parts = [
        f"Hazard: {norm_class.replace('_', ' ').title()} (Severity: {factors.severity}/40).",
        f"Damaged area proxy is {area_ratio*100:.2f}% of visual frame (Area: {factors.area}/20).",
        f"Detector confidence is {clamped_conf:.2f} (Confidence: {factors.confidence}/15).",
        f"Road context is {norm_cat.value.replace('_', ' ').title()} (Road Impact: {factors.road_importance}/25)."
    ]
    if norm_cat == RoadCategory.UNKNOWN:
        explanation_parts.append("Note: Road type unclassified; applied neutral default (15 pts).")

    explanation = " ".join(explanation_parts)

    return SingleRiskAssessment(
        damage_class=norm_class,
        confidence=round(clamped_conf, 3),
        bounding_box=[round(c, 1) for c in bounding_box],
        road_category=norm_cat.value,
        risk_score=final_score,
        priority=priority,
        factors=factors,
        explanation=explanation
    )


def assess_image_risk(
    detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    road_category: Optional[str] = "unknown"
) -> ImageRiskSummary:
    """
    Assess overall road hazard risk across multiple detections on an image.
    
    Prevents risk inflation from multiple or overlapping bounding boxes using
    a non-linear aggregation model:
      Image Risk = Clamped(R_max + 0.15 * sum(R_secondary * (1 - IoU_with_primary)))
    """
    if not detections:
        return ImageRiskSummary(
            highest_hazard=None,
            image_risk_score=0,
            image_priority=PRIORITY_LOW,
            total_detections=0,
            hazard_breakdown={},
            individual_assessments=[],
            summary_explanation="No road hazards detected. Road surface in acceptable condition."
        )

    # Evaluate each detection individually
    individual_assessments: List[SingleRiskAssessment] = []
    hazard_breakdown: Dict[str, int] = {}

    for det in detections:
        d_class = det.get("class_name") or det.get("damage_class") or "pothole"
        conf = float(det.get("score") if det.get("score") is not None else det.get("confidence", 0.5))
        box = det.get("box") or det.get("bounding_box") or [0, 0, 0, 0]

        assessment = assess_detection_risk(
            damage_class=d_class,
            confidence=conf,
            bounding_box=box,
            image_width=image_width,
            image_height=image_height,
            road_category=road_category
        )
        individual_assessments.append(assessment)
        hazard_breakdown[assessment.damage_class] = hazard_breakdown.get(assessment.damage_class, 0) + 1

    # Sort descending by individual risk score
    individual_assessments.sort(key=lambda a: a.risk_score, reverse=True)
    highest_hazard = individual_assessments[0]

    # Aggregate: Primary hazard sets base risk.
    # Secondary hazards contribute diminishing risk discounted by overlap with higher-severity boxes.
    base_score = float(highest_hazard.risk_score)
    marginal_addition = 0.0

    for i in range(1, len(individual_assessments)):
        sec = individual_assessments[i]
        # Check max overlap with any higher-ranked hazard
        max_overlap = max(
            calculate_iou(sec.bounding_box, individual_assessments[j].bounding_box)
            for j in range(i)
        )
        # Overlapping boxes (IoU > 0.4) reflect redundant views of same defect; discount heavily
        discount = max(0.0, 1.0 - max_overlap)
        marginal_addition += 0.15 * sec.risk_score * discount

    aggregated_score = int(round(max(0.0, min(100.0, base_score + marginal_addition))))
    image_priority = get_priority_category(aggregated_score)

    summary_text = (
        f"Analyzed {len(individual_assessments)} hazard(s). Primary hazard is "
        f"{highest_hazard.damage_class.replace('_', ' ')} with individual risk of {highest_hazard.risk_score}/100. "
        f"Overall image risk score is {aggregated_score}/100 ({image_priority} priority)."
    )

    return ImageRiskSummary(
        highest_hazard=highest_hazard,
        image_risk_score=aggregated_score,
        image_priority=image_priority,
        total_detections=len(individual_assessments),
        hazard_breakdown=hazard_breakdown,
        individual_assessments=individual_assessments,
        summary_explanation=summary_text
    )

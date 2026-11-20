"""
RoadSight AI - Risk Service
Adapter connecting inference detections to the Stage 3 Risk Engine.
"""

from typing import List, Dict, Any, Optional
from risk_engine import assess_image_risk, ImageRiskSummary, normalize_road_category


class RiskService:

    @staticmethod
    def evaluate(
        detections: List[Dict[str, Any]],
        image_width: int,
        image_height: int,
        road_category: Optional[str] = "unknown",
        dual_pass_executed: bool = False,
        inference_pass_used: str = "standard"
    ) -> ImageRiskSummary:
        """
        Executes explainable risk analysis across image detections using
        the documented Stage 3 risk engine.
        """
        norm_cat = normalize_road_category(road_category)
        return assess_image_risk(
            detections=detections,
            image_width=image_width,
            image_height=image_height,
            road_category=norm_cat.value,
            dual_pass_executed=dual_pass_executed,
            inference_pass_used=inference_pass_used
        )

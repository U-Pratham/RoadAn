"""
RoadSight AI - Pydantic Request & Response Schemas
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator


class FactorScoresSchema(BaseModel):
    severity: float
    area: float
    confidence: float
    road_importance: float
    total: float


class DetectionItemSchema(BaseModel):
    damage_class: str
    confidence: float
    bounding_box: List[float]
    individual_risk_score: int
    priority_category: str
    explanation: str
    factors: Optional[FactorScoresSchema] = None
    inference_pass: Optional[str] = "standard"
    enhancement_note: Optional[str] = None


class AnalysisResponseSchema(BaseModel):
    analysis_id: int
    original_filename: str
    upload_timestamp: str
    image_dimensions: str
    road_category: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    overall_risk_score: int
    priority_category: str
    summary_explanation: str
    inference_latency_ms: float
    total_processing_ms: float
    detection_count: int
    detections: List[DetectionItemSchema]
    annotated_image_url: str
    dual_pass_executed: Optional[bool] = False
    inference_pass_used: Optional[str] = "standard"
    applied_confidence_threshold: float = 0.20


class AnalysisDetailSchema(BaseModel):
    id: int
    original_filename: str
    upload_timestamp: str
    image_path: str
    annotated_path: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    road_category: str
    overall_risk_score: int
    priority_category: str
    inference_ms: float
    total_ms: float
    summary_explanation: str
    detection_count: int
    detections: List[DetectionItemSchema]
    dual_pass_executed: Optional[bool] = False
    inference_pass_used: Optional[str] = "standard"
    applied_confidence_threshold: float = 0.20


class HistoryItemSchema(BaseModel):
    id: int
    original_filename: str
    upload_timestamp: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    road_category: str
    overall_risk_score: int
    priority_category: str
    detection_count: int
    inference_ms: float


class HistoryResponseSchema(BaseModel):
    total: int
    limit: int
    offset: int
    items: List[HistoryItemSchema]


class StatisticsResponseSchema(BaseModel):
    total_analyses_count: int
    total_detections_count: int
    average_risk_score: float
    average_inference_latency_ms: float
    detections_by_class: Dict[str, int]
    analyses_by_priority: Dict[str, int]
    analyses_by_road_category: Dict[str, int]


class HealthResponseSchema(BaseModel):
    status: str
    model: str
    active_provider: str
    gpu_accelerated: bool
    hardware_target: str

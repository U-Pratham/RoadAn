"""
RoadSight AI - History, Statistics & Health API Routes
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status

from app.schemas import (
    HistoryResponseSchema,
    AnalysisDetailSchema,
    StatisticsResponseSchema,
    HealthResponseSchema,
    DetectionItemSchema
)
from app.database import get_analyses_history, get_analysis_by_id, get_system_statistics
from app.services.inference_service import InferenceService

router = APIRouter(prefix="/api", tags=["history_and_metrics"])


@router.get(
    "/health",
    response_model=HealthResponseSchema,
    summary="Check service and hardware health"
)
async def health_check():
    try:
        infer_service = InferenceService.get_instance()
        provider = infer_service.active_provider
        is_gpu = "Dml" in provider or "CUDA" in provider
        hardware = "NVIDIA GeForce RTX 2050 (DirectML / DX12)" if is_gpu else "CPUExecutionProvider"
    except Exception as e:
        provider = "Error"
        is_gpu = False
        hardware = str(e)

    return HealthResponseSchema(
        status="healthy",
        model="YOLO11n-RDD2022",
        active_provider=provider,
        gpu_accelerated=is_gpu,
        hardware_target=hardware
    )


@router.get(
    "/history",
    response_model=HistoryResponseSchema,
    summary="Get paginated analysis history"
)
async def list_history(
    limit: int = Query(50, ge=1, le=100, description="Items per page"),
    offset: int = Query(0, ge=0, description="Page offset"),
    priority: Optional[str] = Query(None, pattern="^(Critical|High|Moderate|Low)$", description="Filter by priority")
):
    result = get_analyses_history(limit=limit, offset=offset, priority=priority)
    return HistoryResponseSchema(
        total=result["total"],
        limit=result["limit"],
        offset=result["offset"],
        items=result["items"]
    )


@router.get(
    "/history/{analysis_id}",
    response_model=AnalysisDetailSchema,
    summary="Get detailed inspection record"
)
async def get_history_detail(analysis_id: int):
    record = get_analysis_by_id(analysis_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection record with ID {analysis_id} not found."
        )

    detections = []
    for d in record.get("detections", []):
        detections.append(DetectionItemSchema(
            damage_class=d["damage_class"],
            confidence=d["confidence"],
            bounding_box=d["bounding_box"],
            individual_risk_score=d["individual_risk_score"],
            priority_category=d["priority_category"],
            explanation=d["explanation"]
        ))

    return AnalysisDetailSchema(
        id=record["id"],
        original_filename=record["original_filename"],
        upload_timestamp=record["upload_timestamp"],
        image_path=record["image_path"],
        annotated_path=record["annotated_path"],
        latitude=record["latitude"],
        longitude=record["longitude"],
        road_category=record["road_category"],
        overall_risk_score=record["overall_risk_score"],
        priority_category=record["priority_category"],
        inference_ms=record["inference_ms"],
        total_ms=record["total_ms"],
        summary_explanation=record["summary_explanation"],
        detection_count=record["detection_count"],
        detections=detections
    )


@router.get(
    "/statistics",
    response_model=StatisticsResponseSchema,
    summary="Get actual system-wide hazard statistics"
)
async def get_statistics():
    stats = get_system_statistics()
    return StatisticsResponseSchema(
        total_analyses_count=stats["total_analyses_count"],
        total_detections_count=stats["total_detections_count"],
        average_risk_score=stats["average_risk_score"],
        average_inference_latency_ms=stats["average_inference_latency_ms"],
        detections_by_class=stats["detections_by_class"],
        analyses_by_priority=stats["analyses_by_priority"],
        analyses_by_road_category=stats["analyses_by_road_category"]
    )

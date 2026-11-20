"""
RoadSight AI - Detection API Routes
Handles road image upload, validation, inference, and image delivery.
"""

from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, status
from fastapi.responses import FileResponse

from app.schemas import AnalysisResponseSchema, DetectionItemSchema, FactorScoresSchema
from app.services.storage_service import StorageService
from app.services.inference_service import InferenceService
from app.services.risk_service import RiskService
from app.database import save_analysis, get_analysis_by_id
from app.config import DEFAULT_CONFIDENCE_THRESHOLD

router = APIRouter(prefix="/api", tags=["detection"])


@router.post(
    "/detect",
    response_model=AnalysisResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and analyze a road photograph"
)
async def detect_road_damage(
    file: UploadFile = File(..., description="Road photograph (JPEG, PNG, WebP)"),
    confidence_threshold: float = Form(
        DEFAULT_CONFIDENCE_THRESHOLD,
        ge=0.05,
        le=0.95,
        description="Detection confidence threshold"
    ),
    latitude: Optional[float] = Form(
        None,
        ge=-90.0,
        le=90.0,
        description="Optional GPS latitude (-90 to +90)"
    ),
    longitude: Optional[float] = Form(
        None,
        ge=-180.0,
        le=180.0,
        description="Optional GPS longitude (-180 to +180)"
    ),
    road_category: Optional[str] = Form(
        "unknown",
        description="Road category: highway, main_urban, residential, internal, unknown"
    )
):
    # 1. Validate & Save Uploaded Image securely
    saved_img_path, clean_orig_name = await StorageService.validate_and_save_upload(file)

    # 2. Run Single Persistent ONNX Inference Session
    try:
        infer_service = InferenceService.get_instance()
        infer_result = infer_service.predict(
            image_path=saved_img_path,
            conf_thresh=confidence_threshold
        )
    except Exception as e:
        # Cleanup uploaded file on unhandled inference error
        if saved_img_path.exists():
            saved_img_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference engine failure: {str(e)}"
        )

    # 3. Calculate Explainable Risk Assessments
    dual_pass_executed = infer_result.get("dual_pass_executed", False)
    inference_pass_used = infer_result.get("inference_pass_used", "standard")

    risk_summary = RiskService.evaluate(
        detections=infer_result["detections"],
        image_width=infer_result["orig_w"],
        image_height=infer_result["orig_h"],
        road_category=road_category,
        dual_pass_executed=dual_pass_executed,
        inference_pass_used=inference_pass_used
    )

    # 4. Generate & Save Annotated Image
    unique_stem = saved_img_path.stem
    annotated_dest = StorageService.get_annotated_path(unique_stem)
    infer_service.save_annotated(
        img=infer_result["image_cv"],
        detections=infer_result["detections"],
        dest_path=annotated_dest
    )

    # 5. Prepare Detection Records for Relational Persistence
    det_records_for_db = []
    for assessment in risk_summary.individual_assessments:
        det_records_for_db.append({
            "damage_class": assessment.damage_class,
            "confidence": assessment.confidence,
            "bounding_box": assessment.bounding_box,
            "risk_score": assessment.risk_score,
            "priority": assessment.priority,
            "explanation": assessment.explanation,
            "inference_pass": assessment.inference_pass
        })

    # 6. Save Relational Record to SQLite
    try:
        analysis_id = save_analysis(
            original_filename=clean_orig_name,
            image_path=str(saved_img_path),
            annotated_path=str(annotated_dest),
            latitude=latitude,
            longitude=longitude,
            road_category=road_category if road_category else "unknown",
            overall_risk_score=risk_summary.image_risk_score,
            priority_category=risk_summary.image_priority,
            inference_ms=infer_result["inference_ms"],
            total_ms=infer_result["total_ms"],
            summary_explanation=risk_summary.summary_explanation,
            detections=det_records_for_db
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database persistence failure: {str(e)}"
        )

    # 7. Build Pydantic Response
    response_detections = []
    for a in risk_summary.individual_assessments:
        response_detections.append(DetectionItemSchema(
            damage_class=a.damage_class,
            confidence=a.confidence,
            bounding_box=a.bounding_box,
            individual_risk_score=a.risk_score,
            priority_category=a.priority,
            explanation=a.explanation,
            inference_pass=a.inference_pass,
            enhancement_note=a.enhancement_note,
            factors=FactorScoresSchema(
                severity=a.factors.severity,
                area=a.factors.area,
                confidence=a.factors.confidence,
                road_importance=a.factors.road_importance,
                total=a.factors.total
            )
        ))

    return AnalysisResponseSchema(
        analysis_id=analysis_id,
        original_filename=clean_orig_name,
        upload_timestamp=str(annotated_dest.stat().st_mtime), # updated on write
        image_dimensions=f"{infer_result['orig_w']}x{infer_result['orig_h']}",
        road_category=road_category if road_category else "unknown",
        latitude=latitude,
        longitude=longitude,
        overall_risk_score=risk_summary.image_risk_score,
        priority_category=risk_summary.image_priority,
        summary_explanation=risk_summary.summary_explanation,
        inference_latency_ms=infer_result["inference_ms"],
        total_processing_ms=infer_result["total_ms"],
        detection_count=len(response_detections),
        detections=response_detections,
        annotated_image_url=f"/api/images/{analysis_id}",
        dual_pass_executed=dual_pass_executed,
        inference_pass_used=inference_pass_used
    )


@router.get(
    "/images/{analysis_id}",
    summary="Retrieve original or annotated image file"
)
async def get_analysis_image(
    analysis_id: int,
    type: str = Query("annotated", pattern="^(annotated|original)$")
):
    analysis = get_analysis_by_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Analysis with ID {analysis_id} not found.")

    raw_path_str = analysis["annotated_path"] if type == "annotated" else analysis["image_path"]
    target_path = Path(raw_path_str)

    # Cross-platform resolution: if absolute path doesn't exist, search relative to storage dirs
    if not target_path.exists():
        filename = Path(raw_path_str).name
        alt_path = (OUTPUTS_DIR if type == "annotated" else UPLOADS_DIR) / filename
        if alt_path.exists():
            target_path = alt_path
        elif type == "annotated":
            # Fallback to original image if annotated image was not persisted
            orig_name = Path(analysis["image_path"]).name
            orig_alt = UPLOADS_DIR / orig_name
            if orig_alt.exists():
                target_path = orig_alt

    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Image file missing from storage.")

    media_type = "image/jpeg" if target_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
    return FileResponse(path=target_path, media_type=media_type)

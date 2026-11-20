"""
RoadSight AI - FastAPI Application Factory
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes.detection import router as detection_router
from app.routes.history import router as history_router
from app.services.inference_service import InferenceService
from app.config import STATIC_DIR, DB_PATH
from app.database import init_db

# Configure clean logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("roadsight")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event: load ONNX model once and initialize database."""
    logger.info("Initializing RoadSight AI backend...")
    init_db()

    # Pre-warm single persistent inference session
    try:
        service = InferenceService.get_instance()
        logger.info(f"Model successfully loaded. Active provider: {service.active_provider}")
    except Exception as e:
        logger.error(f"Failed to load ONNX inference session: {e}")
        raise e

    yield
    logger.info("RoadSight AI backend shutting down.")


app = FastAPI(
    title="RoadSight AI - Road Damage Detection & Risk Intelligence API",
    description="Explainable computer vision and risk scoring platform for municipal road inspection.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(detection_router)
app.include_router(history_router)

# Global safe exception handler: never leak raw stack traces to clients
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred while processing the request."}
    )

# Static file mount for web dashboard
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "RoadSight AI API is running. Visit /docs for OpenAPI documentation."}

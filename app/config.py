"""
RoadSight AI - Configuration Module
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs" / "web"
STATIC_DIR = BASE_DIR / "static"
TEST_IMAGES_DIR = DATA_DIR / "test_images"

# Create required directories
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Database
DB_PATH = BASE_DIR / "roadsight.db"

# Model Configuration
DEFAULT_MODEL_PATH = WEIGHTS_DIR / "yolo11n.onnx"
DEFAULT_CONFIDENCE_THRESHOLD = 0.20
DEFAULT_IOU_THRESHOLD = 0.45
TARGET_IMAGE_SIZE = 640

# Upload Security & Validation
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp"
}

# Hardware Preference
DEFAULT_DEVICE = os.getenv("DEVICE", "cpu")  # Defaults to CPU on Linux/Render; set DEVICE=gpu for GPU

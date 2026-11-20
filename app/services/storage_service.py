"""
RoadSight AI - Storage & File Security Service
Handles file validation, path traversal prevention, and secure disk persistence.
"""

import os
import uuid
from pathlib import Path
from typing import Tuple
from fastapi import UploadFile, HTTPException
from app.config import (
    UPLOADS_DIR,
    OUTPUTS_DIR,
    MAX_FILE_SIZE_BYTES,
    ALLOWED_EXTENSIONS,
    ALLOWED_CONTENT_TYPES
)


class StorageService:

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Strip directory paths and suspicious characters to prevent path traversal."""
        base = Path(filename).name
        # Keep alphanumeric, underscores, dots, hyphens
        clean = "".join(c for c in base if c.isalnum() or c in (".", "_", "-"))
        return clean if clean else "unnamed_image.jpg"

    @classmethod
    async def validate_and_save_upload(cls, file: UploadFile) -> Tuple[Path, str]:
        """
        Validates content type, file extension, and file size.
        Saves file under a collision-safe UUID path.
        Returns: (saved_file_path, sanitized_original_name)
        """
        orig_name = cls.sanitize_filename(file.filename or "upload.jpg")
        ext = Path(orig_name).suffix.lower()

        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file format '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported media type '{file.content_type}'. Must be a valid JPEG, PNG, or WebP image."
            )

        # Read contents with size guard
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024*1024)} MB."
            )

        if len(contents) == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty (0 bytes)."
            )

        # Generate unique storage path
        safe_stem = f"{uuid.uuid4().hex}_{orig_name}"
        dest_path = UPLOADS_DIR / safe_stem

        with open(dest_path, "wb") as f:
            f.write(contents)

        return dest_path, orig_name

    @staticmethod
    def get_annotated_path(unique_stem: str) -> Path:
        """Generate output path for annotated images."""
        return OUTPUTS_DIR / f"annotated_{unique_stem}.jpg"

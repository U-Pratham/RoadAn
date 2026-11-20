"""
RoadSight AI - Inference Service
Manages single persistent ONNX session lifecycle and executes road damage detection.
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
import onnxruntime as ort

from app.config import DEFAULT_MODEL_PATH, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_IOU_THRESHOLD, DEFAULT_DEVICE
from infer import letterbox, decode_and_nms, draw_detections, get_ort_session


class InferenceService:
    _instance = None

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, device: str = DEFAULT_DEVICE):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        self.model_path = str(model_path)
        self.device = device

        # Load session ONCE at startup
        self.session = get_ort_session(self.model_path, device=self.device)
        self.active_provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Perform a single warmup pass
        dummy_blob = np.zeros((1, 3, 640, 640), dtype=np.float32)
        self.session.run(None, {self.input_name: dummy_blob})

    @classmethod
    def get_instance(cls) -> "InferenceService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def predict(
        self,
        image_path: Path,
        conf_thresh: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_thresh: float = DEFAULT_IOU_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Executes model inference on a saved image path.
        Returns: {
            'image_cv': ndarray,
            'orig_w': int,
            'orig_h': int,
            'inference_ms': float,
            'total_ms': float,
            'detections': List[Dict]
        }
        """
        t_start = time.perf_counter()

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"OpenCV could not decode image: {image_path.name}")

        orig_h, orig_w = img.shape[:2]

        # 1. Preprocess (Letterbox)
        t_pre0 = time.perf_counter()
        blob, scale, px, py = letterbox(img, target_size=640)
        t_pre = (time.perf_counter() - t_pre0) * 1000

        # 2. Run persistent ONNX Session
        t_inf0 = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: blob})
        t_inf = (time.perf_counter() - t_inf0) * 1000

        # 3. Postprocess (Decode & NMS)
        t_post0 = time.perf_counter()
        detections = decode_and_nms(
            outputs[0],
            scale=scale,
            px=px,
            py=py,
            orig_w=orig_w,
            orig_h=orig_h,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh
        )
        t_post = (time.perf_counter() - t_post0) * 1000

        total_time = (time.perf_counter() - t_start) * 1000

        return {
            "image_cv": img,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "inference_ms": round(t_inf, 2),
            "total_ms": round(total_time, 2),
            "detections": detections,
            "provider": self.active_provider
        }

    def save_annotated(self, img: np.ndarray, detections: List[Dict[str, Any]], dest_path: Path) -> Path:
        """Draws bounding boxes and labels and writes output to disk."""
        annotated = draw_detections(img, detections)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest_path), annotated)
        return dest_path

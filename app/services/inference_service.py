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

        # 3. Postprocess (Decode & NMS - Pass 1 Standard Pipeline)
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

        for d in detections:
            d["inference_pass"] = "standard"
            d["enhancement_note"] = None

        pass1_count = len(detections)
        pass2_count = 0
        dual_pass_executed = False
        inference_pass_used = "standard"
        pass1_detections = [dict(d) for d in detections]
        pass2_detections = []

        # Pass 2: Automatic Dual-Pass Fallback (Executed ONLY when Pass 1 yields 0 detections)
        if len(detections) == 0:
            dual_pass_executed = True
            gamma = 1.5
            lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
            img_enh = cv2.LUT(img, lut)

            blob_enh, scale_enh, px_enh, py_enh = letterbox(img_enh, target_size=640)
            t_inf2_start = time.perf_counter()
            outputs_enh = self.session.run(None, {self.input_name: blob_enh})
            t_inf += (time.perf_counter() - t_inf2_start) * 1000

            enhanced_dets = decode_and_nms(
                outputs_enh[0],
                scale=scale_enh,
                px=px_enh,
                py=py_enh,
                orig_w=orig_w,
                orig_h=orig_h,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh
            )
            pass2_count = len(enhanced_dets)
            pass2_detections = [dict(d) for d in enhanced_dets]

            # Validate enhanced detections to avoid degenerate / implausible boxes
            valid_enhanced = []
            for ed in enhanced_dets:
                bx = ed.get("box", [0, 0, 0, 0])
                x1 = max(0.0, min(float(orig_w), float(bx[0])))
                y1 = max(0.0, min(float(orig_h), float(bx[1])))
                x2 = max(0.0, min(float(orig_w), float(bx[2])))
                y2 = max(0.0, min(float(orig_h), float(bx[3])))
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                area = bw * bh
                if bw >= 5 and bh >= 5 and area > 25:
                    ed["box"] = [x1, y1, x2, y2]
                    ed["inference_pass"] = "enhanced_wet_road"
                    ed["enhancement_note"] = "Detected via Wet-Road / Puddle Contrast Boost"
                    valid_enhanced.append(ed)

            if len(valid_enhanced) > 0:
                detections = valid_enhanced
                inference_pass_used = "enhanced_wet_road"

        total_time = (time.perf_counter() - t_start) * 1000

        return {
            "image_cv": img,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "inference_ms": round(t_inf, 2),
            "total_ms": round(total_time, 2),
            "detections": detections,
            "provider": self.active_provider,
            "dual_pass_executed": dual_pass_executed,
            "inference_pass_used": inference_pass_used,
            "pass1_count": pass1_count,
            "pass2_count": pass2_count,
            "audit": {
                "pass1_detections": pass1_detections,
                "pass2_detections": pass2_detections,
                "pass1_count": pass1_count,
                "pass2_count": pass2_count,
                "dual_pass_executed": dual_pass_executed,
                "inference_pass_used": inference_pass_used
            }
        }

    def save_annotated(self, img: np.ndarray, detections: List[Dict[str, Any]], dest_path: Path) -> Path:
        """Draws bounding boxes and labels and writes output to disk."""
        annotated = draw_detections(img, detections)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest_path), annotated)
        return dest_path

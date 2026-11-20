"""
RoadSight AI - Standalone ONNX Inference Engine for Road Damage Detection
Model: YOLO11n fine-tuned on RDD2022 India
Supported Classes:
  0: longitudinal_crack (D00)
  1: transverse_crack   (D10)
  2: alligator_crack    (D20)
  3: pothole            (D40)
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

CLASS_NAMES = [
    "longitudinal_crack",
    "transverse_crack",
    "alligator_crack",
    "pothole"
]

CLASS_COLORS = {
    0: (255, 140, 0),    # Dark Orange (BGR: 0, 140, 255)
    1: (0, 165, 255),    # Orange-Yellow
    2: (255, 0, 200),    # Magenta / Purple
    3: (0, 0, 255),      # Bright Red for Potholes
}

def letterbox(img: np.ndarray, target_size: int = 640):
    """
    Resize and pad image with gray (114, 114, 114) keeping aspect ratio.
    Returns: padded_image blob, scale, pad_x, pad_y
    """
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    px = (target_size - nw) // 2
    py = (target_size - nh) // 2

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    canvas[py:py + nh, px:px + nw] = resized

    # Convert BGR to RGB, normalize to [0, 1], transpose to CHW
    blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))  # (3, 640, 640)
    blob = np.expand_dims(blob, axis=0)   # (1, 3, 640, 640)

    return blob, scale, px, py

def compute_iou(box1, box2):
    """Compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def decode_and_nms(output: np.ndarray, scale: float, px: int, py: int, orig_w: int, orig_h: int, conf_thresh: float = 0.20, iou_thresh: float = 0.45):
    """
    Decode raw YOLO output [1, 8, 8400] and apply multi-class NMS.
    """
    data = output[0]  # shape: (8, 8400)
    num_classes = 4

    cx = data[0]
    cy = data[1]
    bw = data[2]
    bh = data[3]
    scores = data[4:4 + num_classes]  # shape: (4, 8400)

    class_ids = np.argmax(scores, axis=0)  # (8400,)
    best_scores = np.max(scores, axis=0)   # (8400,)

    mask = best_scores >= conf_thresh
    if not np.any(mask):
        return []

    valid_cx = cx[mask]
    valid_cy = cy[mask]
    valid_bw = bw[mask]
    valid_bh = bh[mask]
    valid_scores = best_scores[mask]
    valid_classes = class_ids[mask]

    # Convert from 640x640 letterbox coordinates to original image coordinates
    x1 = (valid_cx - valid_bw / 2.0 - px) / scale
    y1 = (valid_cy - valid_bh / 2.0 - py) / scale
    x2 = (valid_cx + valid_bw / 2.0 - px) / scale
    y2 = (valid_cy + valid_bh / 2.0 - py) / scale

    # Clip to image boundaries
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    # Sort by score descending
    sort_indices = np.argsort(-valid_scores)
    boxes = []
    for idx in sort_indices:
        boxes.append({
            "box": [float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])],
            "score": float(valid_scores[idx]),
            "class_id": int(valid_classes[idx]),
            "class_name": CLASS_NAMES[int(valid_classes[idx])]
        })

    # Class-wise Non-Maximum Suppression
    keep = []
    for candidate in boxes:
        overlap = False
        for chosen in keep:
            if chosen["class_id"] == candidate["class_id"]:
                if compute_iou(chosen["box"], candidate["box"]) > iou_thresh:
                    overlap = True
                    break
        if not overlap:
            keep.append(candidate)

    return keep

def draw_detections(img: np.ndarray, detections: list):
    """Draw bounding boxes and styled labels on image."""
    annotated = img.copy()

    for det in detections:
        box = [int(v) for v in det["box"]]
        cls_id = det["class_id"]
        cls_name = det["class_name"]
        score = det["score"]
        color = CLASS_COLORS.get(cls_id, (0, 255, 0))

        # Draw bounding box
        cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, 2)

        # Label text
        label = f"{cls_name} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Label background
        label_y1 = max(0, box[1] - th - 6)
        label_y2 = box[1]
        cv2.rectangle(annotated, (box[0], label_y1), (box[0] + tw + 6, label_y2), color, -1)

        # Label text
        text_color = (0, 0, 0) if cls_id in (0, 1) else (255, 255, 255)
        cv2.putText(annotated, label, (box[0] + 3, label_y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    return annotated

def get_ort_session(model_path: str, device: str = "cpu"):
    """Initialize ONNX Runtime session with requested device priority."""
    available = ort.get_available_providers()
    if device.lower() in ("gpu", "cuda"):
        providers = [p for p in ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    return ort.InferenceSession(model_path, providers=providers)

def run_inference(image_path: str, model_path: str = "weights/yolo11n.onnx", output_dir: str = "outputs", device: str = "cpu", conf_thresh: float = 0.20):
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_file))
    if img is None:
        raise ValueError(f"Failed to read image with OpenCV: {image_path}")
    orig_h, orig_w = img.shape[:2]

    session = get_ort_session(model_path, device=device)
    active_provider = session.get_providers()[0]

    # Preprocess
    t_pre0 = time.perf_counter()
    blob, scale, px, py = letterbox(img, target_size=640)
    t_pre = (time.perf_counter() - t_pre0) * 1000

    # Model Inference
    input_name = session.get_inputs()[0].name
    t_inf0 = time.perf_counter()
    outputs = session.run(None, {input_name: blob})
    t_inf = (time.perf_counter() - t_inf0) * 1000

    # Postprocess
    t_post0 = time.perf_counter()
    detections = decode_and_nms(outputs[0], scale, px, py, orig_w, orig_h, conf_thresh=conf_thresh)
    t_post = (time.perf_counter() - t_post0) * 1000

    # Annotate Image
    annotated_img = draw_detections(img, detections)

    # Save Output
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"annotated_{image_file.name}"
    cv2.imwrite(str(out_path), annotated_img)

    return {
        "image": str(image_file.name),
        "dimensions": f"{orig_w}x{orig_h}",
        "provider": active_provider,
        "inference_ms": round(t_inf, 2),
        "total_ms": round(t_pre + t_inf + t_post, 2),
        "detections_count": len(detections),
        "detections": detections,
        "output_path": str(out_path)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoadSight AI ONNX Inference")
    parser.add_argument("--image", type=str, default="data/test_images/01.jpg", help="Path to input road image")
    parser.add_argument("--model", type=str, default="weights/yolo11n.onnx", help="Path to ONNX model")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save annotated image")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Inference device")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--all", action="store_true", help="Run on all images in data/test_images")

    args = parser.parse_args()

    if args.all:
        img_dir = Path("data/test_images")
        images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        print(f"Running inference on {len(images)} images using {args.device.upper()} (conf={args.conf})...")
        print("-" * 80)
        for img_p in images:
            res = run_inference(str(img_p), model_path=args.model, output_dir=args.output_dir, device=args.device, conf_thresh=args.conf)
            dets_str = ", ".join([f"{d['class_name']} ({d['score']:.2f})" for d in res["detections"]]) if res["detections"] else "None (Normal/Undamaged Road)"
            print(f"[{res['image']:30s}] Latency: {res['inference_ms']:5.1f} ms | Found ({res['detections_count']}): {dets_str}")
        print("-" * 80)
        print(f"All outputs saved to: {Path(args.output_dir).resolve()}")
    else:
        res = run_inference(args.image, model_path=args.model, output_dir=args.output_dir, device=args.device, conf_thresh=args.conf)
        print(f"=== Inference Report for {res['image']} ===")
        print(f"Active Provider: {res['provider']}")
        print(f"Inference Latency: {res['inference_ms']} ms (Total: {res['total_ms']} ms)")
        print(f"Detections Found: {res['detections_count']}")
        for i, d in enumerate(res["detections"], 1):
            box = [round(b, 1) for b in d["box"]]
            print(f"  {i}. {d['class_name']} (conf: {d['score']:.3f}) at bbox {box}")
        print(f"Saved annotated image to: {res['output_path']}")

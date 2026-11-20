"""
RoadSight AI - Stage 2 Inference Reliability Audit Script
Evaluates road damage detection model across multiple thresholds (0.15, 0.25, 0.40),
profiles latency on GPU/CPU, and saves annotated images into outputs/stage2/.
"""

import os
import sys
import json
import time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

from infer import letterbox, decode_and_nms, draw_detections, get_ort_session

def evaluate_suite():
    model_path = "weights/yolo11n.onnx"
    test_dir = Path("data/test_images")
    output_base = Path("outputs/stage2")
    output_base.mkdir(parents=True, exist_ok=True)

    thresholds = [0.15, 0.25, 0.40]
    for t in thresholds:
        (output_base / f"conf_{int(t*100)}").mkdir(parents=True, exist_ok=True)

    # Gather test images
    all_images = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")))
    print(f"Total images found in {test_dir}: {len(all_images)}")

    # Initialize GPU session (DirectML)
    session_gpu = get_ort_session(model_path, device="gpu")
    provider_gpu = session_gpu.get_providers()[0]

    # Initialize CPU session for comparison
    session_cpu = get_ort_session(model_path, device="cpu")
    provider_cpu = session_cpu.get_providers()[0]

    input_name = session_gpu.get_inputs()[0].name

    results_by_image = {}
    threshold_comparison = {0.15: {"total_dets": 0, "by_class": {}},
                            0.25: {"total_dets": 0, "by_class": {}},
                            0.40: {"total_dets": 0, "by_class": {}}}

    for img_path in all_images:
        img_name = img_path.name
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]

        blob, scale, px, py = letterbox(img, target_size=640)

        # Warmup GPU
        for _ in range(2):
            session_gpu.run(None, {input_name: blob})

        # Measure GPU latency over 3 runs
        lat_gpu_runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            outputs = session_gpu.run(None, {input_name: blob})
            lat_gpu_runs.append((time.perf_counter() - t0) * 1000)
        gpu_lat = float(np.mean(lat_gpu_runs))

        # Measure CPU latency over 2 runs
        lat_cpu_runs = []
        for _ in range(2):
            t0 = time.perf_counter()
            session_cpu.run(None, {input_name: blob})
            lat_cpu_runs.append((time.perf_counter() - t0) * 1000)
        cpu_lat = float(np.mean(lat_cpu_runs))

        raw_output = outputs[0]

        image_eval = {
            "filename": img_name,
            "dimensions": f"{orig_w}x{orig_h}",
            "gpu_provider": provider_gpu,
            "gpu_latency_ms": round(gpu_lat, 2),
            "cpu_latency_ms": round(cpu_lat, 2),
            "thresholds": {}
        }

        for conf in thresholds:
            dets = decode_and_nms(raw_output, scale, px, py, orig_w, orig_h, conf_thresh=conf, iou_thresh=0.45)
            annotated = draw_detections(img, dets)
            out_file = output_base / f"conf_{int(conf*100)}" / f"annotated_{img_name}"
            cv2.imwrite(str(out_file), annotated)

            image_eval["thresholds"][str(conf)] = {
                "count": len(dets),
                "detections": dets,
                "saved_to": str(out_file)
            }

            threshold_comparison[conf]["total_dets"] += len(dets)
            for d in dets:
                cname = d["class_name"]
                threshold_comparison[conf]["by_class"][cname] = threshold_comparison[conf]["by_class"].get(cname, 0) + 1

        results_by_image[img_name] = image_eval

    # Write JSON summary
    summary_data = {
        "provider_gpu": provider_gpu,
        "provider_cpu": provider_cpu,
        "total_images": len(results_by_image),
        "threshold_summary": threshold_comparison,
        "images": results_by_image
    }

    with open(output_base / "stage2_audit_results.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\n=== Stage 2 Evaluation Complete ===")
    print(f"Evaluated {len(results_by_image)} images.")
    for conf, data in threshold_comparison.items():
        print(f"Confidence {conf:.2f}: Total Detections = {data['total_dets']}, Breakdown = {data['by_class']}")

if __name__ == "__main__":
    evaluate_suite()

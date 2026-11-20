import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from infer import letterbox, compute_iou, CLASS_NAMES

def evaluate_image_thresholds(image_path: str, thresholds=[0.15, 0.20, 0.25, 0.35], iou_thresh=0.45):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read {image_path}")
    orig_h, orig_w = img.shape[:2]
    
    blob, scale, px, py = letterbox(img, target_size=640)
    
    session = ort.InferenceSession("weights/yolo11n.onnx", providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    output = outputs[0]  # shape: (1, 8, 8400)
    data = output[0]     # shape: (8, 8400)
    
    total_raw_candidates = data.shape[1]  # 8400
    
    cx = data[0]
    cy = data[1]
    bw = data[2]
    bh = data[3]
    scores = data[4:8]
    class_ids = np.argmax(scores, axis=0)
    best_scores = np.max(scores, axis=0)
    
    print(f"=== Image: {Path(image_path).name} ({orig_w}x{orig_h}) ===")
    print(f"Raw Grid Anchors / Candidates: {total_raw_candidates}")
    print(f"Max raw score in entire image: {np.max(best_scores):.4f}")
    
    results = {}
    for thresh in thresholds:
        mask = best_scores >= thresh
        count_after_conf = int(np.sum(mask))
        
        if not np.any(mask):
            results[thresh] = {
                "before_conf": total_raw_candidates,
                "after_conf": 0,
                "after_nms": 0,
                "detections": []
            }
            continue
            
        valid_cx = cx[mask]
        valid_cy = cy[mask]
        valid_bw = bw[mask]
        valid_bh = bh[mask]
        valid_scores = best_scores[mask]
        valid_classes = class_ids[mask]
        
        x1 = np.clip((valid_cx - valid_bw / 2.0 - px) / scale, 0, orig_w)
        y1 = np.clip((valid_cy - valid_bh / 2.0 - py) / scale, 0, orig_h)
        x2 = np.clip((valid_cx + valid_bw / 2.0 - px) / scale, 0, orig_w)
        y2 = np.clip((valid_cy + valid_bh / 2.0 - py) / scale, 0, orig_h)
        
        sort_indices = np.argsort(-valid_scores)
        boxes = []
        for idx in sort_indices:
            boxes.append({
                "box": [round(float(x1[idx]), 1), round(float(y1[idx]), 1), round(float(x2[idx]), 1), round(float(y2[idx]), 1)],
                "score": round(float(valid_scores[idx]), 4),
                "class_id": int(valid_classes[idx]),
                "class_name": CLASS_NAMES[int(valid_classes[idx])]
            })
            
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
                
        results[thresh] = {
            "before_conf": total_raw_candidates,
            "after_conf": count_after_conf,
            "after_nms": len(keep),
            "detections": keep
        }
        
    for thresh, res in results.items():
        print(f"\n--- Threshold {thresh:.2f} ---")
        print(f"  Before Conf Filter : {res['before_conf']}")
        print(f"  After Conf Filter  : {res['after_conf']}")
        print(f"  After NMS (IoU 0.45): {res['after_nms']}")
        for d in res['detections']:
            print(f"    - {d['class_name']}: score={d['score']:.4f}, box={d['box']}")
            
    return results

if __name__ == "__main__":
    webp_path = "data/uploads/183da72744974aad9beed45d854c6f8b_OIP.webp"
    if Path(webp_path).exists():
        evaluate_image_thresholds(webp_path)
    
    evaluate_image_thresholds("data/test_images/04.jpg")
    evaluate_image_thresholds("data/test_images/07.jpg")

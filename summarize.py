import json
import numpy as np

with open("outputs/stage2/stage2_audit_results.json") as f:
    data = json.load(f)

print("=== HARDWARE LATENCY SUMMARY ===")
print("Active GPU Provider:", data["provider_gpu"])
print("Active CPU Provider:", data["provider_cpu"])
gpu_lats = [v["gpu_latency_ms"] for v in data["images"].values()]
cpu_lats = [v["cpu_latency_ms"] for v in data["images"].values()]
print(f"GPU Latency across 36 images: Mean={np.mean(gpu_lats):.2f} ms, Median={np.median(gpu_lats):.2f} ms, Min={np.min(gpu_lats):.2f} ms, Max={np.max(gpu_lats):.2f} ms")
print(f"CPU Latency across 36 images: Mean={np.mean(cpu_lats):.2f} ms, Median={np.median(cpu_lats):.2f} ms, Min={np.min(cpu_lats):.2f} ms, Max={np.max(cpu_lats):.2f} ms")

print("\n=== PER-IMAGE RESULTS AT CONF = 0.25 ===")
for name, info in sorted(data["images"].items()):
    d15 = info["thresholds"]["0.15"]
    d25 = info["thresholds"]["0.25"]
    d40 = info["thresholds"]["0.4"]
    dets_str = ", ".join([f"{d['class_name']} ({d['score']:.2f})" for d in d25["detections"]]) if d25["detections"] else "None (No hazard)"
    print(f"{name:32s} | GPU: {info['gpu_latency_ms']:5.1f}ms | Conf 0.15: {d15['count']:2d} | Conf 0.25: {d25['count']:2d} | Conf 0.40: {d40['count']:2d} | Dets(0.25): {dets_str}")

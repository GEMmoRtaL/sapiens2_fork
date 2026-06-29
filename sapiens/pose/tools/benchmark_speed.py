#!/usr/bin/env python3
"""Benchmark Sapiens2 Pose model inference speed across multiple model sizes.

Measures: GPU memory, per-image latency (detector / model / end-to-end),
FPS, model parameter count, and file size. Outputs a structured JSON report.
"""

import json
import os
import sys
import time
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from sapiens.pose.datasets import parse_pose_metainfo, UDPHeatmap
from sapiens.pose.models import init_model
from tqdm import tqdm

# Import shared detector/detection functions from original repo
_vis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vis')
if _vis_dir not in sys.path:
    sys.path.insert(0, _vis_dir)
from vis_pose import _detector_cache, _get_detector, _detect_persons  # noqa: E402

# ---------------------------------------------------------------------------
# Model size → config path (relative to sapiens/pose/)
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "0.4b": "configs/keypoints308/shutterstock_goliath_3po/sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-1024x768.py",
    "0.8b": "configs/keypoints308/shutterstock_goliath_3po/sapiens2_0.8b_keypoints308_shutterstock_goliath_3po-1024x768.py",
    "1b": "configs/keypoints308/shutterstock_goliath_3po/sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py",
    "5b": "configs/keypoints308/shutterstock_goliath_3po/sapiens2_5b_keypoints308_shutterstock_goliath_3po-1024x768.py",
}

# ---------------------------------------------------------------------------
# Per-image inference (with timing).  Mirrors vis_pose.process_one_image
# but injects CUDA-event timers around detector, model forward, and total.
# ---------------------------------------------------------------------------
def process_one_image_timed(image: np.ndarray, model, args):
    """Run detection + pose estimation on one image; return (keypoints,
    keypoint_scores, bboxes, timing_dict)."""
    timer = {}

    # --- detector ---
    torch.cuda.synchronize(args.device)
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    bboxes = _detect_persons(image, args)
    t1.record()
    torch.cuda.synchronize(args.device)
    timer["detector_ms"] = t0.elapsed_time(t1)

    # --- pipeline + preprocessor (not timed separately, folded into total) ---
    inputs_list = []
    data_samples_list = []
    for bbox in bboxes:
        data_info = dict(img=image)
        data_info["bbox"] = bbox[None]
        data_info["bbox_score"] = np.ones(1, dtype=np.float32)
        data = model.pipeline(data_info)
        data = model.data_preprocessor(data)
        inputs_list.append(data["inputs"])
        data_samples_list.append(data["data_samples"])

    inputs = torch.cat(inputs_list, dim=0)

    # --- model forward ---
    torch.cuda.synchronize(args.device)
    m0 = torch.cuda.Event(enable_timing=True)
    m1 = torch.cuda.Event(enable_timing=True)
    m0.record()
    with torch.no_grad():
        pred = model(inputs)
        if model.cfg.val_cfg is not None and model.cfg.val_cfg.get("flip_test", False):
            pred_flipped = model(inputs.flip(-1))
            pred_flipped = pred_flipped.flip(-1)
            flip_indices = model.pose_metainfo["flip_indices"]
            assert len(flip_indices) == pred_flipped.shape[1]
            pred_flipped = pred_flipped[:, flip_indices]
            pred = (pred + pred_flipped) / 2.0
    m1.record()
    torch.cuda.synchronize(args.device)
    timer["model_ms"] = m0.elapsed_time(m1)

    # --- decode ---
    pred_np = pred.cpu().numpy()
    keypoints = []
    keypoint_scores = []
    for i, data_samples in enumerate(data_samples_list):
        kpts_i, scores_i = model.codec.decode(pred_np[i])
        input_size = data_samples["meta"]["input_size"]
        bbox_center = data_samples["meta"]["bbox_center"]
        bbox_scale = data_samples["meta"]["bbox_scale"]
        kpts_i = kpts_i / input_size * bbox_scale + bbox_center - 0.5 * bbox_scale
        keypoints.append(kpts_i[0])
        keypoint_scores.append(scores_i[0])

    return keypoints, keypoint_scores, bboxes, timer


# ---------------------------------------------------------------------------
# Load a single model
# ---------------------------------------------------------------------------
def load_pose_model(size: str, checkpoint_root: str, config_root: str,
                    device: str):
    config_rel = MODEL_CONFIGS[size]
    config_path = os.path.join(config_root, config_rel)
    checkpoint_path = os.path.join(checkpoint_root, f"sapiens2_{size}_pose.safetensors")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = init_model(config_path, checkpoint_path, device=device)

    # Attach pose metainfo (same as vis_pose.py)
    num_keypoints = model.cfg.num_keypoints
    if num_keypoints == 308:
        model.pose_metainfo = parse_pose_metainfo(
            dict(from_file=os.path.join(config_root, "configs/_base_/keypoints308.py"))
        )

    # Attach codec
    codec_type = model.cfg.codec.pop("type")
    assert codec_type == "UDPHeatmap", "Only support UDPHeatmap"
    model.codec = UDPHeatmap(**model.cfg.codec)

    return model


# ---------------------------------------------------------------------------
# Collect machine info
# ---------------------------------------------------------------------------
def get_machine_info() -> dict:
    info = {}
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        info["gpu_memory_total_gb"] = round(props.total_memory / (1024 ** 3), 1)
    else:
        info["gpu_name"] = "N/A"
        info["gpu_count"] = 0
        info["gpu_memory_total_gb"] = 0.0
    info["cuda_version"] = torch.version.cuda or "N/A"
    info["python_version"] = sys.version.split()[0]
    return info


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------
def main():
    parser = ArgumentParser(description="Benchmark Sapiens2 Pose inference speed")
    parser.add_argument("--det-checkpoint", required=True,
                        help="Local DETR snapshot directory")
    parser.add_argument("--checkpoint-root", required=True,
                        help="Directory containing pose/*.safetensors")
    parser.add_argument("--input", required=True,
                        help="Directory of test images")
    parser.add_argument("--output", required=True,
                        help="Directory for benchmark report")
    parser.add_argument("--models", default="0.4b,0.8b,1b,5b",
                        help="Comma-separated model sizes to benchmark")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup images")
    parser.add_argument("--measure", type=int, default=100,
                        help="Number of timed images")
    parser.add_argument("--device", default="cuda:0",
                        help="GPU device")
    parser.add_argument("--bbox-thr", type=float, default=0.3,
                        help="Bounding box score threshold")
    parser.add_argument("--nms-thr", type=float, default=0.3,
                        help="IoU threshold for NMS")
    parser.add_argument("--no-visualize", action="store_true", default=True,
                        help="Skip saving visualization images")
    parser.add_argument("--visualize", dest="no_visualize", action="store_false",
                        help="Save visualization images (slower)")

    args = parser.parse_args()
    models_to_test = [m.strip() for m in args.models.split(",")]

    # Validate model names
    for m in models_to_test:
        if m not in MODEL_CONFIGS:
            print(f"Unknown model size: {m}. Available: {list(MODEL_CONFIGS.keys())}")
            sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Collect image list
    if os.path.isdir(args.input):
        image_names = sorted(
            [n for n in os.listdir(args.input)
             if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        )
        image_paths = [os.path.join(args.input, n) for n in image_names]
    else:
        with open(args.input, "r") as f:
            image_paths = [line.strip() for line in f if line.strip()]
        image_names = [os.path.basename(p) for p in image_paths]

    total_images = len(image_paths)
    if total_images < args.warmup + args.measure:
        print(f"Warning: only {total_images} images available, "
              f"need {args.warmup + args.measure}. Adjusting.")
        args.measure = max(1, total_images - args.warmup)

    warmup_paths = image_paths[:args.warmup]
    timed_paths = image_paths[args.warmup:args.warmup + args.measure]

    # Config root = sapiens/pose/ directory
    config_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    machine_info = get_machine_info()
    print(f"\n{'='*60}")
    print(f"Machine: {machine_info['gpu_name']} x{machine_info['gpu_count']} "
          f"({machine_info['gpu_memory_total_gb']} GB each)")
    print(f"CUDA: {machine_info['cuda_version']}  Python: {machine_info['python_version']}")
    print(f"Test images: {len(timed_paths)}  Warmup: {len(warmup_paths)}")
    print(f"{'='*60}\n")

    all_results = []

    for model_size in models_to_test:
        print(f"\n--- Benchmarking sapiens2_{model_size} ---")

        try:
            model = load_pose_model(model_size, args.checkpoint_root, config_root, args.device)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            all_results.append({
                "model": f"sapiens2_{model_size}",
                "status": "skipped",
                "error": str(e),
            })
            continue
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM: sapiens2_{model_size} does not fit in GPU memory")
            all_results.append({
                "model": f"sapiens2_{model_size}",
                "status": "oom",
            })
            continue

        # Parameter count
        n_params = sum(p.numel() for p in model.parameters())
        checkpoint_path = os.path.join(args.checkpoint_root,
                                       f"sapiens2_{model_size}_pose.safetensors")
        file_size_gb = round(os.path.getsize(checkpoint_path) / (1024 ** 3), 2)

        # Warm up detector (once per model — detector is cached, so first call loads it)
        print(f"  Loading DETR detector...")
        _get_detector(args.device, args.det_checkpoint)

        # --- Warmup ---
        print(f"  Warming up ({len(warmup_paths)} images)...")
        for img_path in tqdm(warmup_paths, desc=f"  warmup {model_size}", leave=False):
            image = cv2.imread(img_path)
            try:
                process_one_image_timed(image, model, args)
            except Exception:
                pass  # ignore warmup errors
        torch.cuda.synchronize(args.device)

        # --- Timed measurement ---
        torch.cuda.reset_peak_memory_stats(args.device)
        detector_times = []
        model_times = []
        total_times = []

        print(f"  Measuring ({len(timed_paths)} images)...")
        for img_path in tqdm(timed_paths, desc=f"  bench {model_size}", leave=False):
            image = cv2.imread(img_path)

            torch.cuda.synchronize(args.device)
            t_start = time.perf_counter()
            try:
                _, _, _, timer = process_one_image_timed(
                    image, model, args,
                )
            except Exception as e:
                print(f"  [warn] inference failed on {os.path.basename(img_path)}: {e}")
                continue
            t_end = time.perf_counter()

            detector_times.append(timer["detector_ms"])
            model_times.append(timer["model_ms"])
            total_times.append((t_end - t_start) * 1000)

        peak_mem = torch.cuda.max_memory_allocated(args.device) / (1024 ** 3)

        # Aggregate
        result = {
            "model": f"sapiens2_{model_size}",
            "status": "ok",
            "params_million": round(n_params / 1e6, 1),
            "file_size_gb": file_size_gb,
            "gpu_memory_peak_gb": round(peak_mem, 2),
            "images_measured": len(total_times),
            "total_time_s": round(sum(total_times) / 1000, 2),
            "avg_per_image_ms": round(np.mean(total_times), 2),
            "avg_detector_ms": round(np.mean(detector_times), 2),
            "avg_model_ms": round(np.mean(model_times), 2),
            "fps": round(1000 / np.mean(total_times), 3) if total_times else 0,
        }
        all_results.append(result)

        # Console summary
        print(f"  Params: {result['params_million']}M  "
              f"File: {result['file_size_gb']} GB  "
              f"Peak VRAM: {result['gpu_memory_peak_gb']} GB")
        print(f"  Avg total: {result['avg_per_image_ms']:.1f} ms  "
              f"Detector: {result['avg_detector_ms']:.1f} ms  "
              f"Model: {result['avg_model_ms']:.1f} ms  "
              f"FPS: {result['fps']:.3f}")

        # Clean up before next model
        del model
        _detector_cache.clear()
        torch.cuda.empty_cache()

    # --- Final summary table ---
    print(f"\n{'='*80}")
    print(f"{'Model':<18} {'Params':>8} {'File':>7} {'VRAM':>7} {'Avg/Img':>9} {'Det(ms)':>8} {'Model(ms)':>9} {'FPS':>7}")
    print(f"{'-'*80}")
    for r in all_results:
        if r.get("status") != "ok":
            print(f"{r['model']:<18} {'-- ' + r['status'].upper():>42}")
        else:
            print(f"{r['model']:<18} {r['params_million']:>6.0f}M {r['file_size_gb']:>5.1f}G {r['gpu_memory_peak_gb']:>5.1f}G "
                  f"{r['avg_per_image_ms']:>7.1f}ms {r['avg_detector_ms']:>6.1f}ms {r['avg_model_ms']:>7.1f}ms {r['fps']:>5.3f}")
    print(f"{'='*80}")

    # --- Save JSON report ---
    report = {
        "machine": machine_info,
        "config": {
            "image_count_warmup": len(warmup_paths),
            "image_count_measured": args.measure,
            "input_resolution": "1024x768",
            "dataset": "shutterstock_goliath_3po",
        },
        "results": all_results,
    }
    report_path = os.path.join(args.output, "benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()

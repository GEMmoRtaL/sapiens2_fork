#!/usr/bin/env python3
"""Benchmark inference time vs number of persons per image.

Uses the multi-person dataset to measure how detector time, model time,
and total time scale with the number of detected people.
"""

import json
import os
import sys
from argparse import ArgumentParser

import cv2
import numpy as np
import torch

# Add tools dir to path for benchmark_speed imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Add vis dir to path for original repo imports
_vis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vis')
if _vis_dir not in sys.path:
    sys.path.insert(0, _vis_dir)

from benchmark_speed import (
    MODEL_CONFIGS,
    load_pose_model,
    process_one_image_timed,
)
from vis_pose import _get_detector


def main():
    parser = ArgumentParser(
        description="Benchmark time vs person count for Sapiens2 Pose"
    )
    parser.add_argument("--det-checkpoint", required=True,
                        help="Local DETR snapshot directory")
    parser.add_argument("--checkpoint-root", required=True,
                        help="Directory containing pose/*.safetensors")
    parser.add_argument("--input", required=True,
                        help="Directory of multi-person test images")
    parser.add_argument("--output", required=True,
                        help="Directory for benchmark report")
    parser.add_argument("--models", default="0.4b",
                        help="Comma-separated model sizes")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Number of warmup images (not recorded)")
    parser.add_argument("--repeat", type=int, default=5,
                        help="Repeat each image N times, take average")
    parser.add_argument("--device", default="cuda:0",
                        help="GPU device")
    parser.add_argument("--bbox-thr", type=float, default=0.3)
    parser.add_argument("--nms-thr", type=float, default=0.3)

    args = parser.parse_args()
    models_to_test = [m.strip() for m in args.models.split(",")]
    for m in models_to_test:
        if m not in MODEL_CONFIGS:
            raise SystemExit(f"Unknown model: {m}. Available: {list(MODEL_CONFIGS)}")

    os.makedirs(args.output, exist_ok=True)

    # Collect images
    image_names = sorted(
        n for n in os.listdir(args.input)
        if n.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    image_paths = [os.path.join(args.input, n) for n in image_names]
    print(f"Found {len(image_paths)} images in {args.input}")

    config_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    all_results = []

    for model_size in models_to_test:
        print(f"\n{'='*60}")
        print(f" Model: sapiens2_{model_size}  |  Repeat: {args.repeat}x")
        print(f"{'='*60}")

        model = load_pose_model(model_size, args.checkpoint_root, config_root, args.device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Params: {n_params/1e6:.1f}M")

        # Warm up detector + model
        print(f"Warming up ({args.warmup} images)...")
        _get_detector(args.device, args.det_checkpoint)
        for img_path in image_paths[:args.warmup]:
            image = cv2.imread(img_path)
            try:
                process_one_image_timed(
                    image, model, args,
                )
            except Exception:
                pass
        torch.cuda.synchronize(args.device)

        # Timed measurement
        torch.cuda.reset_peak_memory_stats(args.device)

        print(f"Measuring ({len(image_paths)} images × {args.repeat} repeats)...\n")

        image_records = []

        for img_name, img_path in zip(image_names, image_paths):
            image = cv2.imread(img_path)

            # Run repeat times
            run_total = []
            run_detector = []
            run_model = []
            person_counts = []

            for _ in range(args.repeat):
                torch.cuda.synchronize(args.device)
                try:
                    keypoints, keypoint_scores, bboxes, timer = process_one_image_timed(
                        image, model, args,
                    )
                except Exception as e:
                    print(f"  [warn] {img_name}: {e}")
                    continue
                torch.cuda.synchronize(args.device)

                n_persons = len(bboxes)
                person_counts.append(n_persons)
                run_total.append(timer["detector_ms"] + timer["model_ms"])
                run_detector.append(timer["detector_ms"])
                run_model.append(timer["model_ms"])

            if not run_total:
                image_records.append({
                    "image": img_name, "persons": 0, "error": "all runs failed",
                })
                continue

            n_persons = int(np.mean(person_counts))  # should be constant across runs
            avg_total = np.mean(run_total)
            avg_detector = np.mean(run_detector)
            avg_model = np.mean(run_model)
            per_person = avg_total / n_persons if n_persons > 0 else 0

            image_records.append({
                "image": img_name,
                "persons": n_persons,
                "avg_total_ms": round(avg_total, 2),
                "avg_detector_ms": round(avg_detector, 2),
                "avg_model_ms": round(avg_model, 2),
                "per_person_ms": round(per_person, 2),
            })

            # Print per-image line
            print(f"  {img_name:<16}  persons={n_persons:>2}  "
                  f"det={avg_detector:>6.1f}ms  model={avg_model:>7.1f}ms  "
                  f"total={avg_total:>7.1f}ms  per_person={per_person:>7.1f}ms")

        # ---- Aggregate by person count ----
        groups = {}
        for rec in image_records:
            if rec.get("error"):
                continue
            n = rec["persons"]
            groups.setdefault(n, []).append(rec)

        print(f"\n--- Aggregated by person count ---")
        print(f"{'Persons':>8}  {'Images':>6}  {'Avg Total(ms)':>13}  "
              f"{'Avg Det(ms)':>10}  {'Avg Model(ms)':>13}  {'Per Person(ms)':>15}")
        print("-" * 75)
        for n in sorted(groups):
            recs = groups[n]
            avg_t = np.mean([r["avg_total_ms"] for r in recs])
            avg_d = np.mean([r["avg_detector_ms"] for r in recs])
            avg_m = np.mean([r["avg_model_ms"] for r in recs])
            pp = np.mean([r["per_person_ms"] for r in recs])
            print(f"  {n:>6}  {len(recs):>6}  {avg_t:>11.1f}ms  "
                  f"{avg_d:>8.1f}ms  {avg_m:>11.1f}ms  {pp:>13.1f}ms")

        # ---- Linear scaling analysis ----
        single_person_imgs = [r for r in image_records if r.get("persons") == 1]
        if single_person_imgs:
            baseline_ms = np.mean([r["avg_total_ms"] for r in single_person_imgs])
            print(f"\n--- Linear scaling factor (baseline: 1 person = {baseline_ms:.0f}ms) ---")
            for n in sorted(groups):
                if n == 1:
                    continue
                recs = groups[n]
                avg_t = np.mean([r["avg_total_ms"] for r in recs])
                linear_predict = baseline_ms * n
                ratio = avg_t / linear_predict
                print(f"  {n:>6} persons: actual={avg_t:.0f}ms, "
                      f"linear_predict={linear_predict:.0f}ms, "
                      f"ratio={ratio:.2f} "
                      f"({'sub-linear' if ratio < 1 else 'super-linear'})")

        peak_mem = torch.cuda.max_memory_allocated(args.device) / (1024 ** 3)
        result_entry = {
            "model": f"sapiens2_{model_size}",
            "params_million": round(n_params / 1e6, 1),
            "gpu_memory_peak_gb": round(peak_mem, 2),
            "repeat_per_image": args.repeat,
            "images": image_records,
        }
        all_results.append(result_entry)

        del model
        torch.cuda.empty_cache()

    # ---- Save report ----
    report = {
        "config": {
            "input_dir": args.input,
            "repeat": args.repeat,
            "warmup_images": args.warmup,
            "input_resolution": "1024x768",
        },
        "results": all_results,
    }
    report_path = os.path.join(args.output, "multiperson_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()

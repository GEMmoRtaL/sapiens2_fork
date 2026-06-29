#!/usr/bin/env python3
"""Visualize pose results on multi-person dataset images.

Runs inference on each image, draws detected keypoints + skeletons,
and saves the output. Also prints image resolution and person count.
"""

import json
import os
import sys
from argparse import ArgumentParser

import cv2
import numpy as np
import torch

# Add tools dir to path so we can import benchmark_speed helpers
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
    parser = ArgumentParser()
    parser.add_argument("--det-checkpoint", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", default="0.4b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bbox-thr", type=float, default=0.3)
    parser.add_argument("--nms-thr", type=float, default=0.3)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--thickness", type=int, default=2)
    parser.add_argument("--kpt-thr", type=float, default=0.3)
    args = parser.parse_args()

    models_to_test = [m.strip() for m in args.models.split(",")]
    for m in models_to_test:
        if m not in MODEL_CONFIGS:
            raise SystemExit(f"Unknown model: {m}")

    os.makedirs(args.output, exist_ok=True)

    # Collect images
    image_names = sorted(
        n for n in os.listdir(args.input)
        if n.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    image_paths = [os.path.join(args.input, n) for n in image_names]

    config_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for model_size in models_to_test:
        model = load_pose_model(model_size, args.checkpoint_root, config_root, args.device)
        _get_detector(args.device, args.det_checkpoint)

        print(f"{'Image':<18} {'Resolution':>12} {'Size(MB)':>9} {'Persons':>8}  "
              f"{'Det(ms)':>8} {'Model(ms)':>9} {'Total(ms)':>9}")

        records = []
        for img_name, img_path in zip(image_names, image_paths):
            image = cv2.imread(img_path)
            if image is None:
                print(f"  [skip] {img_name}: cannot read")
                continue

            h, w = image.shape[:2]
            file_mb = round(os.path.getsize(img_path) / (1024 ** 2), 2)

            torch.cuda.synchronize(args.device)
            try:
                keypoints, keypoint_scores, bboxes, timer = process_one_image_timed(
                    image, model, args,
                )
            except Exception as e:
                print(f"  [error] {img_name}: {e}")
                continue
            torch.cuda.synchronize(args.device)

            n_persons = len(bboxes)
            total_ms = timer["detector_ms"] + timer["model_ms"]
            print(f"  {img_name:<18} {w:>4}×{h:<4}     {file_mb:>5.2f} MB   "
                  f"{n_persons:>6}  {timer['detector_ms']:>6.1f}  {timer['model_ms']:>7.1f}  {total_ms:>7.1f}")

            # ---- Draw visualization ----
            # Draw bboxes
            vis = image.copy()
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw keypoints
            for kpts, scores in zip(keypoints, keypoint_scores):
                mask = scores > args.kpt_thr
                for i, (x, y) in enumerate(kpts):
                    if mask[i]:
                        cv2.circle(vis, (int(x), int(y)), args.radius, (0, 0, 255), -1)

            # Draw skeleton
            skeleton = model.pose_metainfo["skeleton_links"]
            for kpts, scores in zip(keypoints, keypoint_scores):
                mask = scores > args.kpt_thr
                for (i, j) in skeleton:
                    if mask[i] and mask[j]:
                        pt1 = (int(kpts[i][0]), int(kpts[i][1]))
                        pt2 = (int(kpts[j][0]), int(kpts[j][1]))
                        cv2.line(vis, pt1, pt2, (255, 0, 0), args.thickness)

            # Add overlay text
            cv2.putText(vis, f"{img_name} | {n_persons} persons | {w}x{h}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            save_path = os.path.join(args.output, f"vis_{img_name}")
            cv2.imwrite(save_path, vis)

            records.append({
                "image": img_name,
                "resolution": f"{w}×{h}",
                "file_size_mb": file_mb,
                "persons": n_persons,
                "detector_ms": round(timer["detector_ms"], 2),
                "model_ms": round(timer["model_ms"], 2),
                "total_ms": round(total_ms, 2),
            })

        # Save metadata
        meta_path = os.path.join(args.output, f"vis_metadata_{model_size}.json")
        with open(meta_path, "w") as f:
            json.dump(records, f, indent=2)

        print(f"\nVisualizations saved to: {args.output}/")
        print(f"Metadata: {meta_path}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

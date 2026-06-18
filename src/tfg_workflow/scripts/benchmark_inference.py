#!/usr/bin/env python3
"""Benchmark YOLO inference speed (ms/frame) for all 9 models.

Runs each model on images from CITIC5 and simulated datasets,
measures mean/std ms per frame after GPU warm-up.

Usage:
    python benchmark_inference.py [--n-images 100] [--warmup 10]
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Models to benchmark ────────────────────────────────────────
YOLO_MODELS = [
    'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt',
    'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt',
    'yolo12n.pt', 'yolo12s.pt', 'yolo12m.pt',
]

# ── Datasets ───────────────────────────────────────────────────
DATASETS = {
    'CITIC5': Path(__file__).parent.parent / 'data' / 'citic5' / 'images',
    'Sim (320x240)': Path(__file__).parent.parent / 'data' / 'sim_dataset' / 'images',
}


def collect_images(directory: Path, n: int) -> list[Path]:
    """Collect up to n image paths from directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in extensions
    )
    if len(images) > n:
        # deterministic sample: evenly spaced
        indices = np.linspace(0, len(images) - 1, n, dtype=int)
        images = [images[i] for i in indices]
    return images


def benchmark_model(
    model_name: str,
    images: list[Path],
    warmup: int = 10,
) -> dict:
    """Benchmark a single model. Returns timing stats."""
    print(f"\n  Loading {model_name}...")
    model = YOLO(model_name)

    # Warm-up runs (not measured)
    warmup_imgs = images[:min(warmup, len(images))]
    for img_path in warmup_imgs:
        model.predict(source=str(img_path), verbose=False)

    # Measured runs
    times_ms = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        t0 = time.perf_counter()
        model.predict(source=img, verbose=False)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_arr = np.array(times_ms)
    return {
        'model': model_name,
        'n_images': len(images),
        'mean_ms': float(np.mean(times_arr)),
        'std_ms': float(np.std(times_arr)),
        'min_ms': float(np.min(times_arr)),
        'max_ms': float(np.max(times_arr)),
        'median_ms': float(np.median(times_arr)),
        'fps': float(1000.0 / np.mean(times_arr)),
    }


def main():
    parser = argparse.ArgumentParser(description="YOLO inference benchmark")
    parser.add_argument(
        '--n-images', type=int, default=100,
        help='Number of images to benchmark per dataset (default: 100)'
    )
    parser.add_argument(
        '--warmup', type=int, default=10,
        help='Number of warm-up inferences before measuring (default: 10)'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("YOLO Inference Benchmark — ms/frame")
    print("=" * 70)

    all_results = {}

    for ds_name, ds_path in DATASETS.items():
        if not ds_path.exists():
            print(f"\n⚠ Dataset not found: {ds_path} — skipping")
            continue

        images = collect_images(ds_path, args.n_images)
        if not images:
            print(f"\n⚠ No images found in {ds_path} — skipping")
            continue

        print(f"\n{'=' * 70}")
        print(f"Dataset: {ds_name} ({len(images)} images)")
        print(f"{'=' * 70}")

        ds_results = []
        for model_name in YOLO_MODELS:
            result = benchmark_model(model_name, images, warmup=args.warmup)
            ds_results.append(result)
            print(
                f"  {model_name:<15} → "
                f"{result['mean_ms']:.1f} ± {result['std_ms']:.1f} ms  "
                f"(median {result['median_ms']:.1f} ms, "
                f"FPS {result['fps']:.1f})"
            )

        all_results[ds_name] = ds_results

    # ── Final summary table ────────────────────────────────────
    print("\n\n" + "=" * 90)
    print("RESUMEN COMPLETO — Tiempo de inferencia")
    print("=" * 90)

    for ds_name, ds_results in all_results.items():
        print(f"\n--- {ds_name} ---")
        print(
            f"{'Modelo':<15} {'ms/frame (mean)':>16} {'± std':>10} "
            f"{'mediana':>10} {'FPS':>8}"
        )
        print("-" * 62)
        for r in ds_results:
            print(
                f"{r['model']:<15} {r['mean_ms']:>13.1f} ms {r['std_ms']:>8.1f} ms "
                f"{r['median_ms']:>8.1f} ms {r['fps']:>7.1f}"
            )

    # ── Markdown table for report ──────────────────────────────
    print("\n\n--- Tabla Markdown (para informe) ---\n")
    for ds_name, ds_results in all_results.items():
        print(f"**{ds_name}:**\n")
        print(
            f"| {'Modelo':<15} | {'ms/frame':>10} | {'± std':>8} "
            f"| {'Mediana':>10} | {'FPS':>6} |"
        )
        print(
            f"|{'-' * 17}|{'-' * 12}|{'-' * 10}|{'-' * 12}|{'-' * 8}|"
        )
        for r in ds_results:
            print(
                f"| {r['model']:<15} | {r['mean_ms']:>8.1f} ms | "
                f"{r['std_ms']:>6.1f} ms | {r['median_ms']:>8.1f} ms | "
                f"{r['fps']:>6.1f} |"
            )
        print()

    # ── Combined table (CITIC5 performance + speed) ────────────
    if 'CITIC5 (real)' in all_results or 'CITIC5' in all_results:
        print("\n--- Tabla combinada: Rendimiento CITIC5 + Velocidad ---\n")

        # CITIC5 results from previous evaluation (hardcoded)
        citic5_perf = {
            'yolov8n.pt': {'mAP': 0.7723, 'F1': 0.9429},
            'yolov8s.pt': {'mAP': 0.8113, 'F1': 0.9423},
            'yolov8m.pt': {'mAP': 0.8777, 'F1': 0.9722},
            'yolo11n.pt': {'mAP': 0.7973, 'F1': 0.9608},
            'yolo11s.pt': {'mAP': 0.8608, 'F1': 0.9792},
            'yolo11m.pt': {'mAP': 0.8824, 'F1': 0.9877},
            'yolo12n.pt': {'mAP': 0.8353, 'F1': 0.9779},
            'yolo12s.pt': {'mAP': 0.8848, 'F1': 0.9950},
            'yolo12m.pt': {'mAP': 0.9016, 'F1': 0.9941},
        }

        # Use whichever dataset key exists
        speed_key = 'CITIC5' if 'CITIC5' in all_results else list(all_results.keys())[0]
        speed_data = {r['model']: r for r in all_results[speed_key]}

        print(
            f"| {'Modelo':<15} | {'mAP@0.5':>8} | {'F1':>6} "
            f"| {'ms/frame':>10} | {'FPS':>6} |"
        )
        print(
            f"|{'-' * 17}|{'-' * 10}|{'-' * 8}|{'-' * 12}|{'-' * 8}|"
        )
        for model_name in YOLO_MODELS:
            perf = citic5_perf.get(model_name, {'mAP': 0, 'F1': 0})
            spd = speed_data.get(model_name, {'mean_ms': 0, 'fps': 0})
            print(
                f"| {model_name:<15} | {perf['mAP']:>8.4f} | {perf['F1']:>6.4f} "
                f"| {spd['mean_ms']:>8.1f} ms | {spd['fps']:>6.1f} |"
            )


if __name__ == '__main__':
    main()
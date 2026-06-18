#!/usr/bin/env python3
# Copyright (c) 2026 TFG Workflow
# Licensed under the Apache License, Version 2.0
#
# benchmark_fps_yolo.py: Mide el rendimiento REAL de inferencia (ms/frame y FPS)
# de YOLOv8n frente a YOLOv8s sobre la GPU NVIDIA RTX 2070, replicando la misma
# llamada de inferencia que usa person_detector.py (classes=[0], device=cuda:0).
#
# NOTA CIENTÍFICA: las runs de telemetría de W&B NO registran el FPS de la GPU,
# por lo que la comparativa de rendimiento de inferencia se obtiene aquí de forma
# offline pero con MEDICIONES REALES sobre la misma tarjeta gráfica del TFG.
#
# Salidas:
#   plots/comparativa_fps_yolo.png   (boxplot comparativo de FPS)
#   plots/fps_benchmark.json         (estadísticas consumidas por analyze_telemetry.py)
#   plots/tabla_fps_yolo.tex         (tabla LaTeX comparativa)
#
# Uso:
#   python3 benchmark_fps_yolo.py [--n-images 150] [--warmup 15]

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(os.path.expanduser("~/tfg_ws"))
DEFAULT_OUT = REPO_ROOT / "plots"

# Modelos comparados (los del pipeline de seguimiento del TFG).
MODELS = [
    {"file": "yolov8n.pt", "label": "YOLOv8n", "color": "#2ca02c"},
    {"file": "yolov8s.pt", "label": "YOLOv8s", "color": "#1f77b4"},
]

# Fuentes de imágenes candidatas (se usa la primera que exista con imágenes).
IMAGE_DIRS = [
    REPO_ROOT / "citic5DataSetVOCSeparados" / "citic5DataSetVOC" / "images",
    REPO_ROOT / "data" / "citic5" / "images",
    REPO_ROOT / "data" / "sim_dataset" / "images",
]


def find_model(file_name: str) -> str:
    """Localiza un .pt en la raíz del repo o en src/tfg_workflow."""
    for cand in [REPO_ROOT / file_name,
                 REPO_ROOT / "src" / "tfg_workflow" / file_name,
                 REPO_ROOT / "src" / "tfg_workflow" / "scripts" / file_name]:
        if cand.is_file():
            return str(cand)
    return file_name  # deja que ultralytics lo descargue si hace falta


def collect_images(n: int) -> list:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for d in IMAGE_DIRS:
        if not d.is_dir():
            continue
        imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in exts)
        if imgs:
            if len(imgs) > n:
                idx = np.linspace(0, len(imgs) - 1, n, dtype=int)
                imgs = [imgs[i] for i in idx]
            print(f"[INFO] Usando {len(imgs)} imágenes de {d}")
            return [str(p) for p in imgs]
    return []


def benchmark_model(model_file: str, images: list, warmup: int) -> dict:
    import cv2
    from ultralytics import YOLO

    path = find_model(model_file)
    print(f"\n[INFO] Cargando {model_file} ({path})...")
    model = YOLO(path)

    # Pre-carga de imágenes a memoria para no medir el coste de disco.
    frames = [cv2.imread(p) for p in images]
    frames = [f for f in frames if f is not None]

    # Calentamiento de la GPU (no medido).
    for f in frames[:min(warmup, len(frames))]:
        model.predict(f, classes=[0], device="cuda:0", verbose=False)

    times_ms = []
    for f in frames:
        t0 = time.perf_counter()
        model.predict(f, classes=[0], device="cuda:0", verbose=False)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times_ms)
    fps = 1000.0 / arr  # FPS instantáneo por frame
    return {
        "model": model_file,
        "n_frames": len(arr),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "median_ms": float(np.median(arr)),
        "mean_fps": float(fps.mean()),
        "std_fps": float(fps.std()),
        "median_fps": float(np.median(fps)),
        "fps_samples": fps.tolist(),
    }


def plot_fps(results: list, out_dir: Path):
    data = [r["fps_samples"] for r in results]
    labels = [next(m["label"] for m in MODELS if m["file"] == r["model"]) for r in results]
    colors = [next(m["color"] for m in MODELS if m["file"] == r["model"]) for r in results]

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    for i, vals in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(vals))
        ax.scatter(x, vals, color=colors[i], s=10, alpha=0.4, zorder=3)
    for i, r in enumerate(results):
        ax.text(i + 1, r["median_fps"], f"  {r['median_fps']:.0f} FPS",
                va="center", fontsize=10, fontweight="bold")

    ax.set_xticklabels([f"{l}\n({d[0] and len(d)} frames)" for l, d in zip(labels, data)], fontsize=10)
    ax.set_ylabel("FPS de inferencia (Hz)", fontsize=11)
    ax.set_title("Rendimiento de inferencia YOLOv8n vs YOLOv8s\nNVIDIA RTX 2070 (8 GB) — medición offline real",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = out_dir / "comparativa_fps_yolo.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[SUCCESS] Gráfica de FPS exportada: {path}")


def export_latex(results: list, out_dir: Path):
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Rendimiento de inferencia medido sobre la GPU NVIDIA RTX 2070 (8\\,GB). "
        "Medición offline real replicando la llamada de inferencia de \\texttt{person\\_detector.py} "
        "(\\texttt{classes=[0]}, \\texttt{device=cuda:0}).}",
        "\\label{tab:fps_yolo}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Modelo & ms/frame (media) & ms/frame (mediana) & FPS (media) & FPS (mediana) \\\\",
        "\\midrule",
    ]
    for r in results:
        label = next(m["label"] for m in MODELS if m["file"] == r["model"])
        lines.append(
            f"    {label} & ${r['mean_ms']:.1f}\\pm{r['std_ms']:.1f}$ & ${r['median_ms']:.1f}$ "
            f"& ${r['mean_fps']:.1f}\\pm{r['std_fps']:.1f}$ & ${r['median_fps']:.1f}$ \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path = out_dir / "tabla_fps_yolo.tex"
    path.write_text("\n".join(lines) + "\n")
    print(f"[SUCCESS] Tabla LaTeX FPS exportada: {path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark real de FPS YOLOv8n vs YOLOv8s en RTX 2070")
    parser.add_argument("--n-images", type=int, default=150)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(args.n_images)
    if not images:
        print("[ERROR] No se encontraron imágenes para el benchmark.")
        return 1

    results = [benchmark_model(m["file"], images, args.warmup) for m in MODELS]

    print("\n" + "=" * 64)
    print("  RESULTADOS BENCHMARK FPS (RTX 2070)")
    print("=" * 64)
    for r in results:
        print(f"  {r['model']:<12} {r['mean_ms']:6.1f} ms/frame  ->  "
              f"{r['mean_fps']:6.1f} FPS (mediana {r['median_fps']:.1f})")

    # Resumen sin las muestras crudas (para el JSON consumido por el redactor).
    summary = {r["model"]: {k: v for k, v in r.items() if k != "fps_samples"} for r in results}
    (out_dir / "fps_benchmark.json").write_text(json.dumps(summary, indent=2))
    print(f"[SUCCESS] JSON de resumen: {out_dir / 'fps_benchmark.json'}")

    plot_fps(results, out_dir)
    export_latex(results, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

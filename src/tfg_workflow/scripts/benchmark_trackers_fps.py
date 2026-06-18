#!/usr/bin/env python3
# Copyright (c) 2026 TFG Workflow
# Licensed under the Apache License, Version 2.0
#
# benchmark_trackers_fps.py: Mide el coste temporal del pipeline de tracking en la
# RTX 2070: tiempo de detección (YOLO11s) y overhead de cada tracker de BoxMOT
# (update por frame). Reporta FPS del tracker (solo asociación) y end-to-end
# (detección + tracking). Genera tabla LaTeX.
#
# Relevante para el robot real: los trackers con ReID (DeepOCSORT, StrongSORT)
# son más precisos en identidad pero más caros (extracción de embeddings osnet).
#
# Uso:
#   python3 benchmark_trackers_fps.py --source data/mot_dataset/seq03/img1

import argparse
import os
import time
from pathlib import Path

import numpy as np

REPO = Path(os.path.expanduser("~/tfg_ws"))
DEFAULT_TRACKERS = ["ocsort", "bytetrack", "botsort", "deepocsort", "strongsort"]


def find_model(name):
    for c in [Path(name), REPO / name, REPO / "src" / "tfg_workflow" / name]:
        if c.is_file():
            return str(c)
    return name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="data/mot_dataset/seq03/img1")
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--trackers", default=",".join(DEFAULT_TRACKERS))
    p.add_argument("--reid_weights", default="osnet_x0_25_msmt17.pt")
    p.add_argument("--out", default=os.path.expanduser("~/tfg_ws/plots/tabla_fps_trackers.tex"))
    args = p.parse_args()

    import cv2
    import torch
    from ultralytics import YOLO
    from boxmot.trackers.tracker_zoo import create_tracker, get_tracker_config, REID_TRACKERS

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(find_model(args.model))
    imgs = sorted(p for p in Path(args.source).iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    frames = [cv2.imread(str(p)) for p in imgs]
    print(f"[INFO] {len(frames)} frames de {args.source}")

    # --- Detección: warm-up + medición ---
    for f in frames[:10]:
        model.predict(f, classes=[0], conf=args.conf, device=device, verbose=False)
    det_ms, dets_all = [], []
    for f in frames:
        t0 = time.perf_counter()
        r = model.predict(f, classes=[0], conf=args.conf, device=device, verbose=False)
        det_ms.append((time.perf_counter() - t0) * 1000)
        d = np.empty((0, 6))
        if len(r) and r[0].boxes is not None and len(r[0].boxes) > 0:
            b = r[0].boxes
            d = np.hstack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy().reshape(-1, 1),
                           b.cls.cpu().numpy().reshape(-1, 1)])
        dets_all.append(d)
    det_mean = float(np.mean(det_ms))
    det_fps = 1000.0 / det_mean
    print(f"\n[YOLO11s] detección: {det_mean:.1f} ms/frame  ->  {det_fps:.0f} FPS")

    rows = []
    for tn in args.trackers.split(","):
        tn = tn.strip()
        reid = Path(args.reid_weights) if tn in REID_TRACKERS else None
        trk = create_tracker(tn, get_tracker_config(tn), reid_weights=reid, device=device, half=False)
        # warm-up
        for i in range(min(10, len(frames))):
            trk.update(dets_all[i], frames[i])
        trk = create_tracker(tn, get_tracker_config(tn), reid_weights=reid, device=device, half=False)
        upd_ms = []
        for d, f in zip(dets_all, frames):
            t0 = time.perf_counter()
            trk.update(d, f)
            upd_ms.append((time.perf_counter() - t0) * 1000)
        um = float(np.mean(upd_ms))
        trk_fps = 1000.0 / um if um > 0 else float("inf")
        e2e_fps = 1000.0 / (det_mean + um)
        rows.append((tn, "sí" if reid else "no", um, trk_fps, e2e_fps))
        print(f"  {tn:11} ReID={'sí' if reid else 'no':2}  update={um:6.2f} ms  "
              f"tracker={trk_fps:7.0f} FPS  end2end(YOLO+trk)={e2e_fps:5.1f} FPS")

    # LaTeX
    L = [
        "% Generado por benchmark_trackers_fps.py (RTX 2070).",
        "\\begin{table}[htbp]", "\\centering",
        f"\\caption{{Coste temporal del pipeline de seguimiento en la GPU NVIDIA RTX~2070. "
        f"Detección YOLO11s: ${det_mean:.1f}$~ms/frame (${det_fps:.0f}$~FPS). Se reporta el "
        f"\\emph{{overhead}} de asociación de cada tracker y el rendimiento end-to-end "
        f"(detección + seguimiento). Los trackers con ReID extraen embeddings de apariencia "
        f"(osnet), de ahí su mayor coste.}}",
        "\\label{tab:fps_trackers}",
        "\\begin{tabular}{lcccc}", "\\toprule",
        "Tracker & ReID & Update (ms) & Tracker (FPS) & End-to-end (FPS) \\\\",
        "\\midrule",
    ]
    for tn, reid, um, tf, ef in rows:
        L.append(f"    {tn} & {reid} & ${um:.2f}$ & ${tf:.0f}$ & ${ef:.1f}$ \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n[SUCCESS] Tabla LaTeX FPS trackers: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

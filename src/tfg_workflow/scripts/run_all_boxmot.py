#!/usr/bin/env python3
# Copyright (c) 2026 TFG Workflow
# Licensed under the Apache License, Version 2.0
#
# run_all_boxmot.py: Ejecuta VARIOS trackers de BoxMOT sobre todas las secuencias
# del dataset, corriendo YOLO11s UNA sola vez por frame (cachea detecciones y las
# alimenta a cada tracker). Mucho más rápido que lanzar un proceso por tracker.
# Salida MOTChallenge por (secuencia, tracker).
#
# Uso:
#   python3 run_all_boxmot.py --dataset data/mot_dataset --out /tmp/mot_eval2 \
#       --trackers ocsort,deepocsort,strongsort

import argparse
import os
import time
from pathlib import Path

import numpy as np

REPO = Path(os.path.expanduser("~/tfg_ws"))


def find_model(name):
    for c in [Path(name), REPO / name, REPO / "src" / "tfg_workflow" / name]:
        if c.is_file():
            return str(c)
    return name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/mot_dataset")
    p.add_argument("--out", default="/tmp/mot_eval2")
    p.add_argument("--trackers", default="ocsort,deepocsort,strongsort")
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--reid_weights", default="osnet_x0_25_msmt17.pt")
    args = p.parse_args()

    import cv2
    import torch
    from ultralytics import YOLO
    from boxmot.trackers.tracker_zoo import create_tracker, get_tracker_config, REID_TRACKERS

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(find_model(args.model))
    tnames = [t.strip() for t in args.trackers.split(",") if t.strip()]
    os.makedirs(args.out, exist_ok=True)

    seqs = sorted(p for p in Path(args.dataset).glob("seq*") if p.is_dir())
    print(f"[INFO] {len(seqs)} secuencias, trackers: {tnames}")

    for s in seqs:
        imgs = sorted(p for p in (s / "img1").iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        # Un tracker nuevo por secuencia (estado independiente).
        trackers = {}
        for tn in tnames:
            reid = Path(args.reid_weights) if tn in REID_TRACKERS else None
            trackers[tn] = create_tracker(tn, get_tracker_config(tn),
                                          reid_weights=reid, device=device, half=False)
        files = {tn: open(os.path.join(args.out, f"{s.name}_{tn}.txt"), "w") for tn in tnames}
        counts = {tn: 0 for tn in tnames}

        t0 = time.time()
        for fr, ip in enumerate(imgs, start=1):
            img = cv2.imread(str(ip))
            res = model.predict(img, classes=[0], conf=args.conf, device=device, verbose=False)
            dets = np.empty((0, 6))
            if len(res) and res[0].boxes is not None and len(res[0].boxes) > 0:
                b = res[0].boxes
                dets = np.hstack([b.xyxy.cpu().numpy(),
                                  b.conf.cpu().numpy().reshape(-1, 1),
                                  b.cls.cpu().numpy().reshape(-1, 1)])
            for tn in tnames:
                for t in trackers[tn].update(dets, img):
                    x1, y1, x2, y2, tid = t[0], t[1], t[2], t[3], int(t[4])
                    conf = t[5] if len(t) > 5 else 1.0
                    files[tn].write(f"{fr},{tid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{conf:.4f},-1,-1,-1\n")
                    counts[tn] += 1
        for f in files.values():
            f.close()
        dt = time.time() - t0
        print(f"  {s.name}: {len(imgs)} frames en {dt:.1f}s  | "
              + "  ".join(f"{tn}={counts[tn]}" for tn in tnames))

    print(f"[SUCCESS] Salidas en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

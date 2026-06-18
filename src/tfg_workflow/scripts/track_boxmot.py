#!/usr/bin/env python3
# Copyright (c) 2026 TFG Workflow
# Licensed under the Apache License, Version 2.0
#
# track_boxmot.py: Ejecuta seguimiento multi-objeto con los trackers de la librería
# BoxMOT (OCSORT, ByteTrack, DeepOCSORT, StrongSORT, BoT-SORT, HybridSORT, ...)
# usando YOLO11s como detector, y vuelca el resultado en formato MOTChallenge.
#
# Amplía la comparativa más allá de los 2 trackers de Ultralytics (track_persons.py),
# replicando el enfoque multi-algoritmo del TFM de referencia. Los trackers con ReID
# (strongsort, deepocsort, botsort...) descargan pesos osnet la primera vez.
#
# Uso:
#   python3 track_boxmot.py --source SEQ/img1 --tracker ocsort --out ocsort.txt
#   python3 track_boxmot.py --source SEQ/img1 --tracker deepocsort --out deep.txt

import argparse
import os
from pathlib import Path

import numpy as np

REPO = Path(os.path.expanduser("~/tfg_ws"))
NON_REID = ["ocsort", "bytetrack", "sfsort"]


def find_model(name):
    for c in [Path(name), REPO / name, REPO / "src" / "tfg_workflow" / name]:
        if c.is_file():
            return str(c)
    return name


def main():
    p = argparse.ArgumentParser(description="Tracking con BoxMOT + YOLO11s")
    p.add_argument("--source", required=True, help="Directorio de imágenes (img1)")
    p.add_argument("--tracker", required=True,
                   help="ocsort|bytetrack|sfsort|deepocsort|strongsort|botsort|hybridsort|boosttrack")
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--out", required=True)
    p.add_argument("--reid_weights", default="osnet_x0_25_msmt17.pt")
    args = p.parse_args()

    import cv2
    import torch
    from ultralytics import YOLO
    from boxmot.trackers.tracker_zoo import create_tracker, get_tracker_config, REID_TRACKERS

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = get_tracker_config(args.tracker)
    reid = Path(args.reid_weights) if args.tracker in REID_TRACKERS else None
    print(f"[INFO] Tracker: {args.tracker} (ReID={'sí' if reid else 'no'})  cfg={cfg}")
    tracker = create_tracker(args.tracker, cfg, reid_weights=reid, device=device, half=False)

    model = YOLO(find_model(args.model))
    imgs = sorted(p for p in Path(args.source).iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"[INFO] {len(imgs)} frames de {args.source}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    n_lines = 0
    with open(args.out, "w") as f:
        for fr, ip in enumerate(imgs, start=1):
            img = cv2.imread(str(ip))
            res = model.predict(img, classes=[0], conf=args.conf, device=device, verbose=False)
            dets = np.empty((0, 6))
            if len(res) and res[0].boxes is not None and len(res[0].boxes) > 0:
                b = res[0].boxes
                xyxy = b.xyxy.cpu().numpy()
                conf = b.conf.cpu().numpy().reshape(-1, 1)
                cls = b.cls.cpu().numpy().reshape(-1, 1)
                dets = np.hstack([xyxy, conf, cls])
            tracks = tracker.update(dets, img)  # Nx8: x1,y1,x2,y2,id,conf,cls,ind
            for t in tracks:
                x1, y1, x2, y2, tid = t[0], t[1], t[2], t[3], int(t[4])
                conf = t[5] if len(t) > 5 else 1.0
                f.write(f"{fr},{tid},{x1:.2f},{y1:.2f},{x2-x1:.2f},{y2-y1:.2f},{conf:.4f},-1,-1,-1\n")
                n_lines += 1

    print(f"[SUCCESS] {len(imgs)} frames, {n_lines} detecciones con ID -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

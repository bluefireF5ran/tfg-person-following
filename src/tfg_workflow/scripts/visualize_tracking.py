#!/usr/bin/env python3
# Copyright (c) 2026 TFG Workflow
# Licensed under the Apache License, Version 2.0
#
# visualize_tracking.py: Dibuja la salida de un tracker (formato MOTChallenge) sobre
# una secuencia de imágenes — cajas coloreadas por ID, etiqueta de ID y estela de
# trayectoria — y escribe un vídeo MP4. Material cualitativo para la memoria
# (visualización de trayectorias y de cambios de identidad/ID-switch).
#
# Uso:
#   python3 visualize_tracking.py --source SEQ/img1 --mot deepocsort.txt --out track.mp4
#   # comparar con GT:  --gt SEQ/gt/gt.txt  (dibuja GT en líneas finas blancas)

import argparse
import os
from collections import defaultdict
from pathlib import Path


def color_for(i):
    import colorsys
    h = (i * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR


def load_mot(path):
    d = defaultdict(list)
    for l in open(path):
        f = l.strip().split(",")
        if len(f) < 6:
            continue
        d[int(f[0])].append((int(f[1]), float(f[2]), float(f[3]), float(f[4]), float(f[5])))
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--mot", required=True, help="Salida del tracker (MOTChallenge)")
    p.add_argument("--gt", default=None, help="GT opcional (se dibuja en blanco fino)")
    p.add_argument("--out", required=True, help="Vídeo MP4 de salida")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--trail", type=int, default=25, help="Frames de estela de trayectoria")
    p.add_argument("--scale", type=float, default=1.0, help="Factor de escala del vídeo (p.ej. 2.0)")
    p.add_argument("--title", default=None, help="Título superpuesto (p.ej. nombre del tracker)")
    args = p.parse_args()

    import cv2
    imgs = sorted(p for p in Path(args.source).iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    tracks = load_mot(args.mot)
    gt = load_mot(args.gt) if args.gt else None
    h0, w0 = cv2.imread(str(imgs[0])).shape[:2]
    s = args.scale
    w, h = int(w0 * s), int(h0 * s)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    def S(v):
        return int(v * s)

    history = defaultdict(list)
    for fr, ip in enumerate(imgs, start=1):
        img = cv2.imread(str(ip))
        if s != 1.0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        if gt is not None:
            for (gid, x, y, bw, bh) in gt.get(fr, []):
                cv2.rectangle(img, (S(x), S(y)), (S(x + bw), S(y + bh)), (255, 255, 255), 1)
        for (tid, x, y, bw, bh) in tracks.get(fr, []):
            c = color_for(tid)
            cv2.rectangle(img, (S(x), S(y)), (S(x + bw), S(y + bh)), c, 2)
            cv2.putText(img, f"ID {tid}", (S(x), max(0, S(y) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
            history[tid].append((S(x + bw / 2), S(y + bh)))
            pts = history[tid][-args.trail:]
            for j in range(1, len(pts)):
                cv2.line(img, pts[j - 1], pts[j], c, 2)
        if args.title:
            cv2.rectangle(img, (0, 0), (w, S(18)), (0, 0, 0), -1)
            cv2.putText(img, f"{args.title}  (frame {fr})", (6, S(13)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(img, f"frame {fr}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        vw.write(img)
    vw.release()
    print(f"[SUCCESS] Vídeo de tracking: {args.out} ({len(imgs)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# Copyright (c) 2026 TFG Workflow
# Licensed under the Apache License, Version 2.0
#
# tune_refine.py: Afina OFFLINE el refinado de cajas del GT simulado, sin tener que
# regenerar el dataset cada vez. Usa lo que guardó `capture_mot_dataset.py --save_depth`
# en una secuencia: img1/*.jpg, depth/*.png (16-bit mm) y meta.jsonl (pose del robot,
# sim_time, intrínsecos por frame). Reconstruye las cajas geométricas y reaplica
# compute_frame_gt con los márgenes que se le pasen, dibujando el resultado para
# comparar parámetros visualmente.
#
# Uso:
#   python3 tune_refine.py --seq data/mot_dataset/seq01 \
#       --sdf ../worlds/warehouse_unified.sdf --box_config ../../../data/actor_box_config.json \
#       --near 0.5 --far 0.20 --out /tmp/tune_n50_f20

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import capture_mot_dataset as C        # noqa: E402
from capture_with_gt import parse_actors_from_sdf  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Afinado offline del refinado de cajas (usa --save_depth)")
    p.add_argument("--seq", required=True, help="Secuencia con img1/, depth/, meta.jsonl")
    p.add_argument("--sdf", required=True)
    p.add_argument("--box_config", default=None)
    p.add_argument("--near", type=float, default=0.5, help="Margen de profundidad hacia la cámara")
    p.add_argument("--far", type=float, default=0.20, help="Margen de profundidad hacia atrás (estricto)")
    p.add_argument("--min_frac", type=float, default=0.10)
    p.add_argument("--ground_z", type=float, default=0.0)
    p.add_argument("--person_height", type=float, default=1.75)
    p.add_argument("--half_width", type=float, default=0.18)
    p.add_argument("--out", required=True, help="Directorio de salida (imágenes con cajas)")
    p.add_argument("--frames", default=None, help="Lista de frames a procesar (coma); por defecto todos")
    args = p.parse_args()

    import cv2
    import numpy as np

    seq = args.seq.rstrip("/")
    actors = parse_actors_from_sdf(args.sdf)
    id_map = C.build_id_map(actors)
    box_config = json.load(open(args.box_config)) if args.box_config and os.path.isfile(args.box_config) else {}
    os.makedirs(args.out, exist_ok=True)

    meta = [json.loads(l) for l in open(os.path.join(seq, "meta.jsonl"))]
    if args.frames:
        want = {int(x) for x in args.frames.split(",")}
        meta = [m for m in meta if m["frame"] in want]

    n_boxes = 0
    for m in meta:
        fr = m["frame"]
        dimg = cv2.imread(os.path.join(seq, "depth", f"{fr:06d}.png"), cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(os.path.join(seq, "img1", f"{fr:06d}.jpg"))
        if dimg is None or rgb is None:
            continue
        depth = dimg.astype(np.float32) / 1000.0          # mm -> m
        depth[depth < 0.05] = np.nan
        cam = m["cam"]; rx, ry, ryaw = m["robot"]; wh = tuple(m["wh"])
        rows = C.compute_frame_gt(actors, id_map, (rx, ry, 0.0), ryaw, cam, m["sim_time"], wh,
                                  depth=depth, depth_refine=True,
                                  refine_margin=args.near, back_margin=args.far,
                                  box_config=box_config, half_width=args.half_width,
                                  person_height=args.person_height, ground_z=args.ground_z)
        for (tid, l, t, w, h) in rows:
            c = C._ID_COLORS[(tid - 1) % len(C._ID_COLORS)]
            cv2.rectangle(rgb, (l, t), (l + w, t + h), c, 2)
            cv2.putText(rgb, f"ID:{tid}", (l, max(0, t - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
            n_boxes += 1
        cv2.imwrite(os.path.join(args.out, f"{fr:06d}.jpg"), rgb)
    print(f"[SUCCESS] near={args.near} far={args.far} min_frac={args.min_frac}: "
          f"{len(meta)} frames, {n_boxes} cajas -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

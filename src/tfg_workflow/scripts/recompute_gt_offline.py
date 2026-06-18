#!/usr/bin/env python3
# Copyright (c) 2026 TFG Workflow
# Licensed under the Apache License, Version 2.0
#
# recompute_gt_offline.py: Recalcula el GT (gt/gt.txt + verify/) de TODO el dataset
# MOT a partir de la profundidad y la pose guardadas por
# `capture_mot_dataset.py --save_depth` (depth/*.png + meta.jsonl por secuencia),
# SIN volver a lanzar Gazebo. Permite afinar el refinado de cajas (silueta del
# actor) cuantas veces haga falta de forma offline (feedback iterativo de Eugenio).
#
# Reutiliza exactamente compute_frame_gt() de capture_mot_dataset, así que el GT
# recomputado es idéntico al que produciría una captura en vivo con esos parámetros.
#
# Uso:
#   python3 recompute_gt_offline.py --dataset data/mot_dataset \
#       --sdf ../worlds/warehouse_unified.sdf \
#       --box_config ../../../data/actor_box_config.json \
#       --near 0.5 --far 0.12

import argparse
import glob
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import capture_mot_dataset as C        # noqa: E402
from capture_with_gt import parse_actors_from_sdf  # noqa: E402


def recompute_seq(seq, actors, id_map, box_config, near, far, half_width,
                  person_height, ground_z, draw):
    import cv2
    import numpy as np
    meta_path = os.path.join(seq, "meta.jsonl")
    if not os.path.isfile(meta_path):
        return None
    meta = [json.loads(l) for l in open(meta_path)]
    lines, n_boxes, wh = [], 0, None
    if draw:
        os.makedirs(os.path.join(seq, "verify"), exist_ok=True)
    for m in meta:
        fr = m["frame"]
        dimg = cv2.imread(os.path.join(seq, "depth", f"{fr:06d}.png"), cv2.IMREAD_UNCHANGED)
        if dimg is None:
            continue
        depth = dimg.astype(np.float32) / 1000.0          # mm -> m
        depth[depth < 0.05] = np.nan
        cam = m["cam"]; rx, ry, ryaw = m["robot"]; wh = tuple(m["wh"])
        rows = C.compute_frame_gt(actors, id_map, (rx, ry, 0.0), ryaw, cam, m["sim_time"], wh,
                                  depth=depth, depth_refine=True,
                                  refine_margin=near, back_margin=far,
                                  box_config=box_config, half_width=half_width,
                                  person_height=person_height, ground_z=ground_z)
        for (tid, l, t, w, h) in rows:
            lines.append(f"{fr},{tid},{l},{t},{w},{h},1,1,1")
            n_boxes += 1
        if draw:
            rgb = cv2.imread(os.path.join(seq, "img1", f"{fr:06d}.jpg"))
            if rgb is not None:
                for (tid, l, t, w, h) in rows:
                    c = C._ID_COLORS[(tid - 1) % len(C._ID_COLORS)]
                    cv2.rectangle(rgb, (l, t), (l + w, t + h), c, 2)
                    cv2.putText(rgb, f"ID:{tid}", (l, max(0, t - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
                cv2.imwrite(os.path.join(seq, "verify", f"{fr:06d}.jpg"), rgb)
    os.makedirs(os.path.join(seq, "gt"), exist_ok=True)
    with open(os.path.join(seq, "gt", "gt.txt"), "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    if wh is not None:
        C.write_seqinfo(__import__("pathlib").Path(seq), os.path.basename(seq),
                        len(meta), 10.0, wh[0], wh[1])
    ids = sorted({int(l.split(",")[1]) for l in lines})
    return len(meta), len(lines), n_boxes, ids


def main():
    p = argparse.ArgumentParser(description="Recomputa el GT del dataset MOT offline (usa --save_depth)")
    p.add_argument("--dataset", required=True, help="Dir con seq*/ (img1, depth, meta.jsonl)")
    p.add_argument("--sdf", required=True)
    p.add_argument("--box_config", default=None)
    p.add_argument("--near", type=float, default=0.5)
    p.add_argument("--far", type=float, default=0.35)
    p.add_argument("--half_width", type=float, default=0.18)
    p.add_argument("--person_height", type=float, default=1.75)
    p.add_argument("--ground_z", type=float, default=0.0)
    p.add_argument("--no_draw", action="store_true", help="No regenerar verify/*.jpg")
    args = p.parse_args()

    actors = parse_actors_from_sdf(args.sdf)
    id_map = C.build_id_map(actors)
    box_config = json.load(open(args.box_config)) if args.box_config and os.path.isfile(args.box_config) else {}
    seqs = sorted(d for d in glob.glob(os.path.join(args.dataset, "*"))
                  if os.path.isfile(os.path.join(d, "meta.jsonl")))
    if not seqs:
        print(f"[ERROR] No hay secuencias con meta.jsonl en {args.dataset}. "
              f"¿Capturaste con --save_depth?")
        return 1
    print(f"[RUN] Recomputando GT offline (near={args.near} far={args.far}) en {len(seqs)} secuencias")
    tot = 0
    for seq in seqs:
        r = recompute_seq(seq, actors, id_map, box_config, args.near, args.far,
                          args.half_width, args.person_height, args.ground_z,
                          draw=not args.no_draw)
        if r is None:
            continue
        nf, nrows, nb, ids = r
        tot += nrows
        print(f"  {os.path.basename(seq)}: {nf} frames, {nrows} filas GT, IDs={ids}")
    print(f"[SUCCESS] GT recomputado: {tot} filas en total -> {args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

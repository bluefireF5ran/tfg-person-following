#!/usr/bin/env python3
# Copyright (c) 2026 TFG Workflow
# Licensed under the Apache License, Version 2.0
#
# evaluate_mot_euclid.py: Evaluación CLEAR MOT siguiendo el protocolo del TFM de
# referencia / el tutor (ROBOT7), que NO usa IoU sino DISTANCIA EUCLÍDEA entre
# centros, con umbral por objeto igual a la SEMIDIAGONAL de la caja GT:
#     umbral_i = 0.5 * sqrt(w_i^2 + h_i^2)
# Reporta (py-motmetrics):
#   - TP (num_matches), FP (num_false_positives), MISS/FN (num_misses),
#     MM/IDsw (num_switches)
#   - MOTA en VALOR ABSOLUTO (0..1; mejor = 1; puede ser negativo)
#   - MOTP = distancia media en PÍXELES entre centros emparejados (menor = mejor)
#
# Uso:
#   python3 evaluate_mot_euclid.py --gt robot7/Estatico3Persona/COCO/mot.txt \
#       --pred "DeepSort=robot7/Estatico3Persona/yolov8x/DeepSort/mot.txt"

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import evaluate_mot as E          # noqa: E402  (load_mot + shim numpy para motmetrics)
import motmetrics as mm           # noqa: E402
import numpy as np                # noqa: E402


def evaluate_pair_euclid(gt, pred):
    """Acumula emparejamientos por distancia euclídea entre centros, con umbral
    = semidiagonal de la caja GT (protocolo CLEAR MOT del TFM de referencia)."""
    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(set(gt.index.get_level_values(0)) | set(pred.index.get_level_values(0)))
    for fr in frames:
        gt_f = gt.loc[[fr]] if fr in gt.index.get_level_values(0) else gt.iloc[0:0]
        pr_f = pred.loc[[fr]] if fr in pred.index.get_level_values(0) else pred.iloc[0:0]
        gt_ids = [i[1] for i in gt_f.index]
        pr_ids = [i[1] for i in pr_f.index]

        if len(gt_ids) and len(pr_ids):
            g = gt_f[["X", "Y", "Width", "Height"]].values.astype(float)
            p = pr_f[["X", "Y", "Width", "Height"]].values.astype(float)
            gc = np.stack([g[:, 0] + g[:, 2] / 2, g[:, 1] + g[:, 3] / 2], axis=1)  # centros GT
            pc = np.stack([p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2], axis=1)  # centros pred
            # distancia euclídea Ng x Np
            D = np.linalg.norm(gc[:, None, :] - pc[None, :, :], axis=2)
            # umbral por GT = semidiagonal de su caja
            thr = 0.5 * np.sqrt(g[:, 2] ** 2 + g[:, 3] ** 2)
            D[D > thr[:, None]] = np.nan
        else:
            D = np.empty((len(gt_ids), len(pr_ids)))
        acc.update(gt_ids, pr_ids, D)
    return acc


def main():
    p = argparse.ArgumentParser(description="Evaluación MOT euclídea (protocolo ROBOT7/TFM)")
    p.add_argument("--gt", required=True)
    p.add_argument("--pred", action="append", required=True, help="'Nombre=fichero.txt' (repetible)")
    p.add_argument("--out", default=None, help="Tabla LaTeX opcional")
    args = p.parse_args()

    gt = E.load_mot(args.gt)
    preds = E.parse_pred_arg(args.pred)
    mh = mm.metrics.create()
    METRICS = ["num_objects", "num_matches", "num_false_positives", "num_misses",
               "num_switches", "mota", "motp", "num_unique_objects"]

    accs, names = [], []
    for name, path in preds.items():
        print(f"[INFO] {name}: {path}")
        accs.append(evaluate_pair_euclid(gt, E.load_mot(path)))
        names.append(name)
    summ = mh.compute_many(accs, names=names, metrics=METRICS, generate_overall=False)

    print(f"\n  {'Tracker':16} {'TP':>5} {'FP':>4} {'MISS':>5} {'MM':>4} {'MOTA':>7} {'MOTP(px)':>9}")
    for n, r in summ.iterrows():
        print(f"  {n:16} {int(r['num_matches']):5d} {int(r['num_false_positives']):4d} "
              f"{int(r['num_misses']):5d} {int(r['num_switches']):4d} "
              f"{r['mota']:7.3f} {r['motp']:9.2f}")
    print("\n  (MOTA absoluto: 1=perfecto, puede ser negativo. MOTP en píxeles: menor=mejor.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

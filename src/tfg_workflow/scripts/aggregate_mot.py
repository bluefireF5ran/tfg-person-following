#!/usr/bin/env python3
# Copyright (c) 2026 TFG Workflow
# Licensed under the Apache License, Version 2.0
#
# aggregate_mot.py: Agrega la evaluación CLEAR MOT de varios trackers sobre TODAS
# las secuencias del dataset (a IoU>=0.5 estándar y IoU>=0.3 laxo) y genera la
# tabla LaTeX comparativa para la memoria. Reutiliza evaluate_mot (incluye el shim
# de numpy para motmetrics).
#
# Uso:
#   python3 aggregate_mot.py --dataset data/mot_dataset --pred_dir /tmp/mot_eval2 \
#       --trackers "ByteTrack=bytetrack,BoT-SORT=botsort,OCSORT=ocsort,DeepOCSORT=deepocsort,StrongSORT=strongsort" \
#       --out plots/tabla_mot_simulado.tex

import argparse
import glob
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import evaluate_mot as E          # noqa: E402  (aplica el shim de numpy)
import motmetrics as mm           # noqa: E402


def aggregate(seqs, pred_dir, suffix, min_iou):
    mh = mm.metrics.create()
    per = []
    for s in seqs:
        gt = E.load_mot(f"{s}/gt/gt.txt", filter_valid=True, ped_only=True)
        pred = E.load_mot(os.path.join(pred_dir, f"{os.path.basename(s)}_{suffix}.txt"))
        per.append(E.evaluate_pair(gt, pred, min_iou))
    ov = mh.compute_many(per, metrics=["mota", "motp", "idf1", "num_switches", "num_fragmentations"],
                         names=[os.path.basename(s) for s in seqs], generate_overall=True).loc["OVERALL"]
    return {"mota": ov["mota"] * 100, "motp": 1 - ov["motp"], "idf1": ov["idf1"] * 100,
            "idsw": int(ov["num_switches"]), "frag": int(ov["num_fragmentations"])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/mot_dataset")
    p.add_argument("--pred_dir", default="/tmp/mot_eval2")
    p.add_argument("--trackers",
                   default="ByteTrack=bytetrack,BoT-SORT=botsort,OCSORT=ocsort,"
                           "DeepOCSORT=deepocsort,StrongSORT=strongsort")
    p.add_argument("--out", default=os.path.expanduser("~/tfg_ws/plots/tabla_mot_simulado.tex"))
    args = p.parse_args()

    seqs = sorted(glob.glob(os.path.join(args.dataset, "seq*")))
    pairs = [t.split("=") for t in args.trackers.split(",")]
    res = {}
    for name, suf in pairs:
        res[name] = {0.5: aggregate(seqs, args.pred_dir, suf, 0.5),
                     0.3: aggregate(seqs, args.pred_dir, suf, 0.3)}
        r5, r3 = res[name][0.5], res[name][0.3]
        print(f"{name:14} IoU0.5: MOTA={r5['mota']:6.1f} IDF1={r5['idf1']:5.1f} IDsw={r5['idsw']:3d} | "
              f"IoU0.3: MOTA={r3['mota']:6.1f} IDF1={r3['idf1']:5.1f}")

    # LaTeX
    L = [
        "% Generado por aggregate_mot.py (agregado de las secuencias del dataset simulado).",
        "\\begin{table}[htbp]", "\\centering",
        "\\caption{Evaluación CLEAR MOT (py-motmetrics) de cinco algoritmos de seguimiento "
        "con detector YOLO11s sobre el dataset simulado etiquetado (5 secuencias, 1000 frames). "
        "Se reportan el umbral estándar MOTChallenge ($\\text{IoU}\\geq0.5$) y uno laxo "
        "($\\text{IoU}\\geq0.3$). MOTA e IDF1 en \\% (mayor mejor); MOTP $=1-\\overline{d}_{IoU}$ "
        "(mayor mejor); IDsw = cambios de identidad (menor mejor).}",
        "\\label{tab:mot_simulado}",
        "\\begin{tabular}{lcccc|cccc}", "\\toprule",
        "& \\multicolumn{4}{c|}{$\\text{IoU}\\geq0.5$} & \\multicolumn{4}{c}{$\\text{IoU}\\geq0.3$} \\\\",
        "Tracker & MOTA & MOTP & IDF1 & IDsw & MOTA & MOTP & IDF1 & IDsw \\\\",
        "\\midrule",
    ]
    for name, _ in pairs:
        a, b = res[name][0.5], res[name][0.3]
        L.append(f"    {name} & ${a['mota']:.1f}$ & ${a['motp']:.3f}$ & ${a['idf1']:.1f}$ & ${a['idsw']}$ "
                 f"& ${b['mota']:.1f}$ & ${b['motp']:.3f}$ & ${b['idf1']:.1f}$ & ${b['idsw']}$ \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n[SUCCESS] Tabla LaTeX: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# Copyright (c) 2026 TFG Workflow
# Licensed under the Apache License, Version 2.0
#
# aggregate_robot7.py: Agrega la evaluación CLEAR MOT (protocolo euclídeo del TFM:
# distancia entre centros + umbral semidiagonal, MOTA absoluto, MOTP en píxeles)
# de varios trackers sobre las secuencias de ROBOT7, y genera una tabla LaTeX.
#
# Espera: robot7/<seq>/gt.txt (robot7_make_gt.py) y predicciones en
#         <pred_dir>/seq_<seq>_<tracker>.txt (run_all_boxmot.py sobre seq*/img1).
#
# Uso:
#   python3 aggregate_robot7.py --pred_dir /tmp/r7_eval --out plots/tabla_mot_robot7.tex

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import evaluate_mot as E
import evaluate_mot_euclid as EU
import motmetrics as mm

SEQS = ["Estatico1Pesona", "Estatico2Persona", "Estatico3Persona", "Seguimiento2Persona"]
SHORT = {"Estatico1Pesona": "Estatico-1p", "Estatico2Persona": "Estatico-2p",
         "Estatico3Persona": "Estatico-3p", "Seguimiento2Persona": "Seguimiento-2p"}
MET = ["num_matches", "num_false_positives", "num_misses", "num_switches", "mota", "motp"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot7", default="robot7")
    p.add_argument("--pred_dir", default="/tmp/r7_eval")
    p.add_argument("--trackers", default="ocsort,bytetrack,botsort,deepocsort,strongsort")
    p.add_argument("--out", default=os.path.expanduser("~/tfg_ws/plots/tabla_mot_robot7.tex"))
    args = p.parse_args()
    trackers = args.trackers.split(",")
    mh = mm.metrics.create()

    # acumulador global (TP,FP,MISS,MM,sum_motp_dist,sum_matches)
    glob = {t: [0, 0, 0, 0, 0.0, 0] for t in trackers}
    per_seq = {}
    for sd in SEQS:
        gt = E.load_mot(os.path.join(args.robot7, sd, "gt.txt"))
        per_seq[sd] = {}
        for t in trackers:
            pf = os.path.join(args.pred_dir, f"seq_{sd}_{t}.txt")
            acc = EU.evaluate_pair_euclid(gt, E.load_mot(pf))
            r = mh.compute(acc, metrics=MET, name=t).loc[t]
            tp, fp, ms, mm_ = (int(r["num_matches"]), int(r["num_false_positives"]),
                               int(r["num_misses"]), int(r["num_switches"]))
            per_seq[sd][t] = (tp, fp, ms, mm_, r["mota"], r["motp"])
            g = glob[t]
            g[0] += tp; g[1] += fp; g[2] += ms; g[3] += mm_
            g[4] += r["motp"] * tp; g[5] += tp

    # consola
    for sd in SEQS:
        print(f"\n== {SHORT[sd]} ==")
        for t, (tp, fp, ms, mm_, mota, motp) in sorted(per_seq[sd].items(), key=lambda x: -x[1][4]):
            print(f"  {t:11} TP={tp:5d} FP={fp:3d} MISS={ms:4d} MM={mm_:3d} MOTA={mota:6.3f} MOTP={motp:5.2f}px")

    # LaTeX (global)
    L = ["% Generado por aggregate_robot7.py (protocolo euclideo TFM, IoU no; umbral semidiagonal).",
         "\\begin{table}[htbp]\n\\centering",
         "\\caption{Evaluación CLEAR MOT sobre ROBOT7 (4 secuencias reales, detector YOLO11s), "
         "siguiendo el protocolo del TFM: emparejamiento por distancia euclídea entre centros con "
         "umbral igual a la semidiagonal de la caja anotada. MOTA en valor absoluto (1=mejor); "
         "MOTP = error medio de localización en píxeles (menor=mejor); TP/FP/MISS/MM agregados de "
         "las 4 secuencias.}",
         "\\label{tab:mot_robot7}",
         "\\begin{tabular}{lccccc c}", "\\toprule",
         "Tracker & TP & FP & MISS & MM (IDsw) & MOTA & MOTP (px) \\\\", "\\midrule"]
    tot_gt = sum(len(E.load_mot(os.path.join(args.robot7, s, "gt.txt"))) for s in SEQS)
    rows = []
    for t in trackers:
        tp, fp, ms, mm_, sdist, sm = glob[t]
        mota = 1 - (fp + ms + mm_) / tot_gt
        motp = sdist / sm if sm else float("nan")
        rows.append((t, tp, fp, ms, mm_, mota, motp))
    for t, tp, fp, ms, mm_, mota, motp in sorted(rows, key=lambda x: -x[5]):
        L.append(f"    {t} & ${tp}$ & ${fp}$ & ${ms}$ & ${mm_}$ & ${mota:.3f}$ & ${motp:.2f}$ \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w").write("\n".join(L) + "\n")
    print(f"\n[SUCCESS] Tabla LaTeX (global) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

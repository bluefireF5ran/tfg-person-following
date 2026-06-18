#!/usr/bin/env python3
# Copyright (c) 2026 TFG Workflow
# Licensed under the Apache License, Version 2.0
#
# aggregate_mot_euclid.py: Agrega la evaluación CLEAR MOT con el protocolo EUCLÍDEO
# del TFM (distancia entre centros + umbral semidiagonal de la caja GT; MOTA
# absoluto; MOTP en píxeles) de varios trackers sobre TODAS las secuencias del
# dataset simulado, y genera la tabla LaTeX comparativa para la memoria.
#
# Es el equivalente para el dataset simulado de aggregate_robot7.py. Espera:
#   <dataset>/seq*/gt/gt.txt   (GT MOTChallenge)
#   <pred_dir>/<seq>_<tracker>.txt   (salidas de run_all_boxmot.py)
#
# Uso:
#   python3 aggregate_mot_euclid.py --dataset data/mot_dataset --pred_dir /tmp/mot_eval_final \
#       --trackers "ByteTrack=bytetrack,BoT-SORT=botsort,OCSORT=ocsort,DeepOCSORT=deepocsort,StrongSORT=strongsort" \
#       --out plots/tabla_mot_simulado_euclid.tex

import argparse
import glob
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import evaluate_mot as E          # noqa: E402  (shim de numpy para motmetrics)
import evaluate_mot_euclid as EU  # noqa: E402
import motmetrics as mm           # noqa: E402

MET = ["num_matches", "num_false_positives", "num_misses", "num_switches", "mota", "motp"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/mot_dataset")
    p.add_argument("--pred_dir", default="/tmp/mot_eval_final")
    p.add_argument("--trackers",
                   default="ByteTrack=bytetrack,BoT-SORT=botsort,OCSORT=ocsort,"
                           "DeepOCSORT=deepocsort,StrongSORT=strongsort")
    p.add_argument("--out", default=os.path.expanduser("~/tfg_ws/plots/tabla_mot_simulado_euclid.tex"))
    p.add_argument("--res", default="640x480", help="Resolución (solo para el pie de tabla)")
    args = p.parse_args()

    # "Display=key" -> (display, key)
    pairs = [(x.split("=")[0], x.split("=")[1]) for x in args.trackers.split(",")]
    seqs = sorted(d for d in glob.glob(os.path.join(args.dataset, "seq*")) if os.path.isdir(d))
    mh = mm.metrics.create()

    glob_acc = {disp: [0, 0, 0, 0, 0.0, 0] for disp, _ in pairs}   # TP,FP,MISS,MM,sum_motp*TP,sum_TP
    per_seq = {}
    tot_gt = 0
    for s in seqs:
        sn = os.path.basename(s)
        gt = E.load_mot(os.path.join(s, "gt", "gt.txt"))
        tot_gt += len(gt) if hasattr(gt, "__len__") else sum(len(v) for v in gt.values())
        per_seq[sn] = {}
        for disp, key in pairs:
            pf = os.path.join(args.pred_dir, f"{sn}_{key}.txt")
            acc = EU.evaluate_pair_euclid(gt, E.load_mot(pf))
            r = mh.compute(acc, metrics=MET, name=disp).loc[disp]
            tp, fp, ms, mm_ = (int(r["num_matches"]), int(r["num_false_positives"]),
                               int(r["num_misses"]), int(r["num_switches"]))
            per_seq[sn][disp] = (tp, fp, ms, mm_, r["mota"], r["motp"])
            g = glob_acc[disp]
            g[0] += tp; g[1] += fp; g[2] += ms; g[3] += mm_
            if r["motp"] == r["motp"]:      # no NaN
                g[4] += r["motp"] * tp; g[5] += tp

    # consola (por secuencia)
    for sn in sorted(per_seq):
        print(f"\n== {sn} ==")
        for disp, (tp, fp, ms, mm_, mota, motp) in sorted(per_seq[sn].items(), key=lambda x: -x[1][4]):
            print(f"  {disp:11} TP={tp:5d} FP={fp:4d} MISS={ms:4d} MM={mm_:3d} MOTA={mota:6.3f} MOTP={motp:5.2f}px")

    L = ["% Generado por aggregate_mot_euclid.py (protocolo euclideo TFM; umbral semidiagonal; IoU no).",
         f"% Dataset simulado (GT refinado por profundidad, imagenes {args.res}). 5 trackers, YOLO11s.",
         "\\begin{table}[htbp]\n\\centering",
         "\\caption{Evaluación CLEAR MOT sobre el dataset simulado (GT refinado por profundidad, "
         f"imágenes ${args.res.replace('x', r'\times ')}$), con el mismo protocolo que ROBOT7: "
         "emparejamiento por distancia euclídea entre centros con umbral igual a la semidiagonal de "
         "la caja anotada. MOTA en valor absoluto (1=mejor); MOTP = error medio de localización en "
         "píxeles (menor=mejor). Detector YOLO11s.}",
         "\\label{tab:mot_simulado_euclid}",
         "\\begin{tabular}{lccccc c}", "\\toprule",
         "Tracker & TP & FP & MISS & MM (IDsw) & MOTA & MOTP (px) \\\\", "\\midrule"]
    rows = []
    for disp, _ in pairs:
        tp, fp, ms, mm_, sdist, sm = glob_acc[disp]
        mota = 1 - (fp + ms + mm_) / tot_gt if tot_gt else float("nan")
        motp = sdist / sm if sm else float("nan")
        rows.append((disp, tp, fp, ms, mm_, mota, motp))
    for disp, tp, fp, ms, mm_, mota, motp in sorted(rows, key=lambda x: -x[5]):
        L.append(f"    {disp} & ${tp}$ & ${fp}$ & ${ms}$ & ${mm_}$ & ${mota:.3f}$ & ${motp:.2f}$ \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w").write("\n".join(L) + "\n")

    print("\n== GLOBAL (5 secuencias) ==")
    for disp, tp, fp, ms, mm_, mota, motp in sorted(rows, key=lambda x: -x[5]):
        print(f"  {disp:11} TP={tp:5d} FP={fp:4d} MISS={ms:4d} MM={mm_:3d} MOTA={mota:6.3f} MOTP={motp:5.2f}px")
    print(f"  (GT total={tot_gt} filas)")
    print(f"\n[SUCCESS] Tabla LaTeX -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# Copyright (c) 2026 TFG Workflow
# Licensed under the Apache License, Version 2.0
#
# evaluate_mot.py: Evalúa el rendimiento de uno o varios algoritmos de seguimiento
# multi-objeto comparando su salida (formato MOTChallenge) contra el ground-truth
# del dataset (p.ej. ROBOT7) usando la biblioteca py-motmetrics. Calcula las
# métricas CLEAR MOT (MOTA, MOTP), IDF1, ID switches, etc., y genera una tabla
# LaTeX comparativa para la memoria del TFG.
#
# Métricas (según el TFM de referencia, sección 4.x y correo del tutor):
#   - MOTA : Multiple Object Tracking Accuracy (mayor es mejor).
#   - MOTP : Multiple Object Tracking Precision (aquí, 1 - dist. media; mayor mejor).
#   - IDF1 : F1 de identidad (consistencia de los IDs; mayor es mejor).
#   - IDsw : nº de cambios de identidad (menor es mejor).
#
# Uso (un tracker):
#   python3 evaluate_mot.py --gt gt.txt --pred bytetrack.txt --name ByteTrack
#
# Uso (comparativa de varios trackers contra el mismo gt):
#   python3 evaluate_mot.py --gt gt.txt \
#       --pred ByteTrack=bytetrack.txt --pred BoT-SORT=botsort.txt \
#       --out plots

import argparse
import os
import sys

try:
    import numpy as np
    # Shim: numpy>=2.0 eliminó np.asfarray, que motmetrics 1.4 aún usa. Lo restauramos
    # antes de importar motmetrics (boxmot fuerza numpy 2.x en este entorno).
    if not hasattr(np, "asfarray"):
        np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
    import motmetrics as mm
    import pandas as pd
except ImportError as e:
    print(f"[ERROR] Falta una dependencia ({e}). Instala: pip install motmetrics pandas numpy")
    sys.exit(1)

# Umbral IoU de emparejamiento. El estándar MOTChallenge es 0.5; el artículo de
# Towards AI (py-motmetrics) usa 0.7. Se deja configurable por CLI (--iou).
DEFAULT_IOU = 0.5


def load_mot(path: str, filter_valid: bool = False, ped_only: bool = False) -> pd.DataFrame:
    """Carga un fichero MOTChallenge: frame,id,bb_left,bb_top,w,h,[flag],[clase],[vis].

    - filter_valid: si el GT trae el flag de "consider" (col. índice 6, formato MOT),
      descarta las filas con flag 0 (cajas a ignorar). Equivale al filtro
      ``line.split[6]==1`` del artículo de referencia.
    - ped_only: si existe columna de clase (índice 7), conserva solo clase 1 (peatón).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, header=None)
    ncols = df.shape[1]

    if filter_valid and ncols >= 7:
        df = df[df.iloc[:, 6] != 0]
    if ped_only and ncols >= 8:
        df = df[df.iloc[:, 7] == 1]

    df = df.iloc[:, :6].copy()
    df.columns = ["FrameId", "Id", "X", "Y", "Width", "Height"]
    df["FrameId"] = df["FrameId"].astype(int)
    df["Id"] = df["Id"].astype(int)
    return df.set_index(["FrameId", "Id"])


def evaluate_pair(gt: pd.DataFrame, pred: pd.DataFrame, min_iou: float) -> mm.MOTAccumulator:
    """Acumula los emparejamientos gt/pred frame a frame por IoU.

    `min_iou` es el IoU MÍNIMO para considerar match (convención MOTChallenge).
    OJO: py-motmetrics `iou_matrix(max_iou=D)` usa D como DISTANCIA máxima (1-IoU),
    por lo que para exigir IoU>=min_iou hay que pasar max_iou = 1 - min_iou.
    """
    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(set(gt.index.get_level_values(0)) |
                    set(pred.index.get_level_values(0)))
    for fr in frames:
        gt_f = gt.loc[fr] if fr in gt.index.get_level_values(0) else pd.DataFrame(columns=["X", "Y", "Width", "Height"])
        pr_f = pred.loc[fr] if fr in pred.index.get_level_values(0) else pd.DataFrame(columns=["X", "Y", "Width", "Height"])

        gt_ids = list(gt_f.index)
        pr_ids = list(pr_f.index)

        if len(gt_ids) and len(pr_ids):
            dists = mm.distances.iou_matrix(
                gt_f[["X", "Y", "Width", "Height"]].values,
                pr_f[["X", "Y", "Width", "Height"]].values,
                max_iou=1.0 - min_iou,
            )
        else:
            dists = np.empty((len(gt_ids), len(pr_ids)))

        acc.update(gt_ids, pr_ids, dists)
    return acc


METRICS = ["num_objects", "num_matches", "mota", "motp", "idf1",
           "num_switches", "num_false_positives", "num_misses",
           "mostly_tracked", "partially_tracked", "mostly_lost",
           "num_fragmentations"]


def export_latex(summary: pd.DataFrame, out_dir: str):
    """Tabla LaTeX comparativa de trackers. MOTP se expresa como 1-dist (mayor mejor)."""
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Evaluación CLEAR MOT de los algoritmos de seguimiento sobre el "
        "dataset ROBOT7, calculada con \\texttt{py-motmetrics}. MOTA e IDF1 en "
        "porcentaje (mayor es mejor); MOTP expresado como $1-\\overline{d}_{IoU}$ "
        "(mayor es mejor); IDsw es el número de cambios de identidad (menor es mejor).}",
        "\\label{tab:mot_robot7}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Tracker & MOTA (\\%) & MOTP & IDF1 (\\%) & IDsw & Frag. \\\\",
        "\\midrule",
    ]
    for name, row in summary.iterrows():
        motp_score = 1.0 - row["motp"] if not np.isnan(row["motp"]) else float("nan")
        lines.append(
            f"    {name} "
            f"& ${row['mota']*100:.1f}$ "
            f"& ${motp_score:.3f}$ "
            f"& ${row['idf1']*100:.1f}$ "
            f"& ${int(row['num_switches'])}$ "
            f"& ${int(row['num_fragmentations'])}$ \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path = os.path.join(out_dir, "tabla_mot_robot7.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[SUCCESS] Tabla LaTeX MOT exportada: {path}")
    print("\n".join(lines))


def parse_pred_arg(values: list) -> dict:
    """Acepta '--pred archivo.txt' o '--pred Nombre=archivo.txt' (varios)."""
    out = {}
    for i, v in enumerate(values):
        if "=" in v:
            name, path = v.split("=", 1)
        else:
            name, path = f"Tracker{i+1}", v
        out[name.strip()] = path.strip()
    return out


def main():
    p = argparse.ArgumentParser(description="Evaluación MOT (CLEAR MOT / py-motmetrics)")
    p.add_argument("--gt", required=True, help="Ground-truth MOTChallenge (frame,id,x,y,w,h,...)")
    p.add_argument("--pred", action="append", required=True,
                   help="Predicción del tracker. Formato: 'Nombre=archivo.txt' (repetible)")
    p.add_argument("--out", default=os.path.expanduser("~/tfg_ws/plots"),
                   help="Directorio de salida para la tabla LaTeX")
    p.add_argument("--iou", type=float, default=DEFAULT_IOU,
                   help=f"IoU MÍNIMO para match (def: {DEFAULT_IOU}, estándar MOTChallenge; usa 0.3 para vista laxa)")
    p.add_argument("--filter-valid", action="store_true",
                   help="Descarta filas del GT con flag de 'consider'=0 (formato MOTChallenge)")
    p.add_argument("--ped-only", action="store_true",
                   help="Conserva solo la clase 1 (peatón) del GT si trae columna de clase")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    gt = load_mot(args.gt, filter_valid=args.filter_valid, ped_only=args.ped_only)
    preds = parse_pred_arg(args.pred)

    accs, names = [], []
    for name, path in preds.items():
        print(f"[INFO] Evaluando {name}: {path}  (IoU={args.iou})")
        pred = load_mot(path)
        accs.append(evaluate_pair(gt, pred, args.iou))
        names.append(name)

    mh = mm.metrics.create()
    summary = mh.compute_many(accs, names=names, metrics=METRICS, generate_overall=False)

    print("\n" + mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap={k: k for k in METRICS},
    ))

    export_latex(summary, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

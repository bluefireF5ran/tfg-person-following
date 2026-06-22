#!/usr/bin/env python3
# aggregate_campaign_offline.py — Lee las ejecuciones OFFLINE de W&B (wandb/offline-run-*) de la
# campaña de seguimiento directamente del datastore local (sin red ni sync) y calcula las métricas
# agregadas (media ± desv.), reutilizando la lógica de analyze_telemetry.py. NO escribe tablas:
# solo imprime, para decidir la integración en la memoria.
#
# Uso:  python3 aggregate_campaign_offline.py [--glob 'wandb/offline-run-20260622_*'] [--min-follow 50]

import argparse, glob, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_telemetry import _coerce_numeric, compute_run_metrics  # misma lógica que la memoria

from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb


def read_offline_run(wandb_file: str) -> pd.DataFrame:
    ds = datastore.DataStore()
    ds.open_for_scan(wandb_file)
    rows = []
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            break
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") == "history":
            row = {}
            for it in rec.history.item:
                key = it.key or ".".join(it.nested_key)
                try:
                    row[key] = json.loads(it.value_json)  # json de Python acepta Infinity/NaN
                except Exception:
                    row[key] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="+", default=["wandb/offline-run-20260622_*"],
                    help="uno o varios patrones glob de carpetas offline (se unen y deduplican)")
    ap.add_argument("--min-follow", type=int, default=50,
                    help="descartar runs con menos de N muestras en FOLLOWING (arranques fallidos)")
    ap.add_argument("--success-thr", type=float, default=0.70,
                    help="umbral de tracking_ratio para considerar una run 'exitosa'")
    ap.add_argument("--plot", default="",
                    help="si se indica una ruta .png, guarda el boxplot de las runs exitosas")
    args = ap.parse_args()

    run_dirs = sorted({d for g in args.glob for d in glob.glob(g)})
    print(f"Encontradas {len(run_dirs)} carpetas offline para {args.glob}\n")

    metrics, skipped = [], 0
    for d in run_dirs:
        wfs = glob.glob(os.path.join(d, "*.wandb"))
        if not wfs:
            continue
        df = _coerce_numeric(read_offline_run(wfs[0]))
        if df.empty or "state_code" not in df.columns:
            skipped += 1
            continue
        n_follow = int((df["state_code"] == 2.0).sum())
        if n_follow < args.min_follow:
            skipped += 1
            continue
        m = compute_run_metrics(df)
        m["run"] = os.path.basename(d).split("-")[-1]
        metrics.append(m)

    n = len(metrics)
    print(f"Runs válidas: {n}  (descartadas {skipped} por vacías/arranque fallido)\n")
    if not n:
        return

    keys = [("mae_ang", "MAE ang (norm)"), ("mae_dist", "MAE dist (m)"),
            ("mean_linear", "v media (m/s)"), ("mean_angular", "w media (rad/s)"),
            ("tracking_ratio", "Seguim (frac)"), ("hit_rate", "Hit rate (%)"),
            ("n_losses", "Perdidas"), ("mean_search_duration_s", "t_SEARCH (s)"),
            ("runtime_s", "Duracion (s)")]

    print("=== AGREGADO (media ± desv. sobre %d runs) ===" % n)
    for k, lab in keys:
        vals = np.array([m[k] for m in metrics], float)
        print(f"  {lab:22s}: {np.nanmean(vals):8.3f} ± {np.nanstd(vals):6.3f}   "
              f"[min {np.nanmin(vals):.3f}, max {np.nanmax(vals):.3f}]")

    # --- Tasa de éxito (nivel-run) y agregado sobre el subconjunto representativo ---
    thr = args.success_thr
    ok = [m for m in metrics if m["tracking_ratio"] >= thr]
    fail = [m for m in metrics if m["tracking_ratio"] < thr]
    print(f"\n=== TASA DE ÉXITO (Seguim >= {thr*100:.0f}%) ===")
    print(f"  Éxito: {len(ok)}/{n}  ({100*len(ok)/n:.1f}%)   |   Fallo (oclusión/robo/pérdida): {len(fail)}")
    print(f"  Runs fallidas: " + ", ".join(f"{m['run']}({m['tracking_ratio']*100:.0f}%)" for m in fail))

    print(f"\n=== PRECISIÓN sobre las {len(ok)} runs exitosas (media ± desv.) ===")
    for k, lab in keys:
        vals = np.array([m[k] for m in ok], float)
        print(f"  {lab:22s}: {np.nanmean(vals):8.3f} ± {np.nanstd(vals):6.3f}")

    if args.plot:
        panels = [("mae_ang", "MAE centrado $e_{ang}$ (norm.)"),
                  ("mae_dist", "MAE distancia (m)"),
                  ("hit_rate", "Tiempo en seguimiento (%)"),
                  ("mean_linear", "$\\bar{v}$ (m/s)"),
                  ("mean_angular", "$\\bar{\\omega}$ (rad/s)")]
        fig, axes = plt.subplots(1, len(panels), figsize=(16, 4.2))
        fig.suptitle(f"Campaña experimental del comportamiento de seguimiento "
                     f"(YOLO11s, {len(ok)} ejecuciones)", fontsize=13, fontweight="bold")
        for ax, (k, title) in zip(axes, panels):
            vals = [m[k] for m in ok]
            bp = ax.boxplot([vals], patch_artist=True, widths=0.5)
            bp["boxes"][0].set_facecolor("#1f77b4"); bp["boxes"][0].set_alpha(0.6)
            x = np.random.normal(1, 0.04, size=len(vals))
            ax.scatter(x, vals, color="#1f77b4", s=18, zorder=3, alpha=0.7)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xticks([]); ax.grid(True, ls="--", alpha=0.4)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(args.plot, dpi=300)
        print(f"\n[plot] boxplot guardado en {args.plot}")


if __name__ == "__main__":
    main()

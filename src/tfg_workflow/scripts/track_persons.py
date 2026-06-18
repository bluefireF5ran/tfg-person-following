#!/usr/bin/env python3
# Copyright (c) 2026 TFG Workflow
# Licensed under the Apache License, Version 2.0
#
# track_persons.py: Ejecuta seguimiento multi-objeto (tracking) de personas sobre
# una secuencia de imágenes o un vídeo, combinando el detector YOLO11s con un
# algoritmo de seguimiento de Ultralytics (ByteTrack o BoT-SORT). Vuelca el
# resultado en formato MOTChallenge para su posterior evaluación con py-motmetrics.
#
# Esta es la "fase de tracking" pedida por el tutor (correo Re: Solucion acceso W&B):
# diferenciar cada persona con una etiqueta/ID estable a lo largo del vídeo.
#   https://docs.ultralytics.com/modes/track
#
# Formato de salida (MOTChallenge, una línea por detección):
#   frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1
#
# Uso:
#   python3 track_persons.py --source SEQ_DIR --tracker bytetrack --out out.txt
#   python3 track_persons.py --source video.mp4 --tracker botsort --model yolo11s.pt

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(os.path.expanduser("~/tfg_ws"))

# Mapeo nombre corto -> fichero de configuración de tracker de Ultralytics.
TRACKERS = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
}


def find_model(name: str) -> str:
    for cand in [Path(name), REPO_ROOT / name,
                 REPO_ROOT / "src" / "tfg_workflow" / name,
                 REPO_ROOT / "src" / "tfg_workflow" / "scripts" / name]:
        if cand.is_file():
            return str(cand)
    return name


def _make_reid_tracker_cfg(base_cfg: str, out_dir: str) -> str:
    """Crea una copia de botsort.yaml con with_reid=True (matching por apariencia).

    Útil cuando las personas son visualmente distintas (en este TFG los actores se
    diferencian por color), lo que mejora la reidentificación tras oclusiones.
    Solo BoT-SORT soporta ReID en Ultralytics.
    """
    import ultralytics
    src = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "trackers", base_cfg)
    lines = []
    seen_reid = False
    with open(src) as f:
        for ln in f:
            if ln.strip().startswith("with_reid:"):
                lines.append("with_reid: True\n")
                seen_reid = True
            else:
                lines.append(ln)
    if not seen_reid:
        lines.append("with_reid: True\n")
    dst = os.path.join(out_dir, "botsort_reid.yaml")
    with open(dst, "w") as f:
        f.writelines(lines)
    return dst


def run_tracking(source: str, model_name: str, tracker: str,
                 conf: float, out_path: str, imgsz: int, reid: bool) -> int:
    from ultralytics import YOLO

    tracker_cfg = TRACKERS.get(tracker, tracker)
    if reid:
        if "botsort" not in tracker_cfg:
            print("[WARN] ReID solo lo soporta BoT-SORT; se ignora --reid.")
        else:
            tracker_cfg = _make_reid_tracker_cfg(tracker_cfg,
                                                 os.path.dirname(os.path.abspath(out_path)) or ".")
            print(f"[INFO] ReID activado (with_reid=True) -> {tracker_cfg}")

    model_path = find_model(model_name)
    print(f"[INFO] Modelo   : {model_path}")
    print(f"[INFO] Tracker  : {tracker} ({tracker_cfg})")
    print(f"[INFO] Fuente   : {source}")

    model = YOLO(model_path)

    # persist=True mantiene los IDs entre frames; stream=True para no agotar RAM.
    results = model.track(
        source=source,
        tracker=tracker_cfg,
        classes=[0],            # solo personas
        conf=conf,
        imgsz=imgsz,
        persist=True,
        stream=True,
        verbose=False,
        device="cuda:0",
    )

    n_lines = 0
    n_frames = 0
    with open(out_path, "w") as f:
        for frame_idx, r in enumerate(results, start=1):
            n_frames += 1
            if r.boxes is None or r.boxes.id is None:
                continue
            ids = r.boxes.id.cpu().numpy().astype(int)
            xywh = r.boxes.xywh.cpu().numpy()      # centro x, centro y, w, h
            confs = r.boxes.conf.cpu().numpy()
            for (cx, cy, w, h), tid, c in zip(xywh, ids, confs):
                bb_left = cx - w / 2.0
                bb_top = cy - h / 2.0
                f.write(f"{frame_idx},{tid},{bb_left:.2f},{bb_top:.2f},"
                        f"{w:.2f},{h:.2f},{c:.4f},-1,-1,-1\n")
                n_lines += 1

    print(f"[SUCCESS] {n_frames} frames procesados, {n_lines} detecciones con ID.")
    print(f"[SUCCESS] Resultados MOTChallenge en: {out_path}")
    return 0


def main():
    p = argparse.ArgumentParser(description="Tracking multi-objeto de personas (YOLO11s + Ultralytics)")
    p.add_argument("--source", required=True, help="Directorio de imágenes o fichero de vídeo")
    p.add_argument("--model", default="yolo11s.pt", help="Modelo detector (def: yolo11s.pt)")
    p.add_argument("--tracker", default="bytetrack", choices=list(TRACKERS) + ["bytetrack.yaml", "botsort.yaml"],
                   help="Algoritmo de seguimiento (def: bytetrack)")
    p.add_argument("--conf", type=float, default=0.3, help="Umbral de confianza")
    p.add_argument("--imgsz", type=int, default=640, help="Tamaño de inferencia")
    p.add_argument("--reid", action="store_true",
                   help="Activa ReID por apariencia (solo BoT-SORT); útil con personas visualmente distintas")
    p.add_argument("--out", required=True, help="Fichero de salida MOTChallenge (.txt)")
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    return run_tracking(args.source, args.model, args.tracker,
                        args.conf, args.out, args.imgsz, args.reid)


if __name__ == "__main__":
    raise SystemExit(main())

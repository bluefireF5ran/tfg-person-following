#!/usr/bin/env python3
# Copyright (c) 2026 TFG Workflow
# Licensed under the Apache License, Version 2.0
#
# robot7_make_gt.py: Genera el ground-truth en formato MOTChallenge (<seq>/gt.txt)
# para una secuencia de ROBOT7, eligiendo la mejor fuente de anotación disponible:
#   1) <seq>/COCO/mot.txt provisto (si trae IDs reales, p.ej. Estatico3).
#   2) CVAT/annotations.xml con identidad en el atributo 'name' (p.ej. Estatico2).
#   3) COCO/labels.json con identidad en attributes.name (p.ej. Estatico1, Seguimiento2).
# El nº de frame se asigna por orden alfabético del nombre de imagen (1..N),
# igual que track_boxmot.py, para alinear GT y predicción.
#
# Uso:  python3 robot7_make_gt.py --seq robot7/Estatico2Persona

import argparse
import glob
import json
import os
import xml.etree.ElementTree as ET


def frame_map(seq):
    imgs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(seq, "*.jpg")))
    return {fn: i + 1 for i, fn in enumerate(imgs)}


def from_cvat(seq, fmap):
    root = ET.parse(os.path.join(seq, "CVAT", "annotations.xml")).getroot()
    rows = []
    names = set()
    for im in root.findall(".//image"):
        fr = fmap.get(im.get("name"))
        if fr is None:
            continue
        for b in im.findall("box"):
            nm = next((a.text for a in b.findall("attribute") if a.get("name") == "name"), None)
            if nm is None:
                continue
            names.add(nm)
            xtl, ytl = float(b.get("xtl")), float(b.get("ytl"))
            xbr, ybr = float(b.get("xbr")), float(b.get("ybr"))
            rows.append([fr, nm, xtl, ytl, xbr - xtl, ybr - ytl])
    return rows, sorted(names)


def from_coco(seq, fmap):
    d = json.load(open(os.path.join(seq, "COCO", "labels.json")))
    images = {im["id"]: im["file_name"] for im in d["images"]}
    rows, names = [], set()
    for a in d["annotations"]:
        fr = fmap.get(images[a["image_id"]])
        nm = a.get("attributes", {}).get("name")
        if fr is None or nm is None:
            continue
        names.add(nm)
        x, y, w, h = a["bbox"]
        rows.append([fr, nm, x, y, w, h])
    return rows, sorted(names)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", required=True, help="Directorio de la secuencia ROBOT7")
    args = p.parse_args()
    seq = args.seq.rstrip("/")
    fmap = frame_map(seq)
    out = os.path.join(seq, "gt.txt")

    # 1) mot.txt provisto con IDs reales (pocos IDs => identidad real, no fallback)
    prov = os.path.join(seq, "COCO", "mot.txt")
    if os.path.isfile(prov):
        ids = {l.split(",")[1] for l in open(prov)}
        if len(ids) <= 12:   # IDs reales de personas, no miles
            with open(out, "w") as f:
                f.write(open(prov).read())
            print(f"[SUCCESS] GT desde COCO/mot.txt provisto (IDs={sorted(ids)}) -> {out}")
            return 0

    # 2) CVAT con nombres
    rows, names = ([], [])
    if os.path.isfile(os.path.join(seq, "CVAT", "annotations.xml")):
        rows, names = from_cvat(seq, fmap)
    # 3) COCO con nombres (si CVAT no aportó)
    if not rows:
        rows, names = from_coco(seq, fmap)

    if not rows:
        print(f"[ERROR] No se encontró identidad en CVAT ni COCO para {seq}")
        return 1
    nid = {n: i + 1 for i, n in enumerate(names)}
    rows.sort(key=lambda r: (r[0], nid[r[1]]))
    with open(out, "w") as f:
        for fr, nm, x, y, w, h in rows:
            f.write(f"{fr},{nid[nm]},{x:.0f},{y:.0f},{w:.0f},{h:.0f},1,-1,-1,-1\n")
    print(f"[SUCCESS] {len(rows)} cajas, identidades {nid} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

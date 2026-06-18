#!/usr/bin/env python3
# Copyright (c) 2026 TFG Workflow
# Licensed under the Apache License, Version 2.0
#
# coco_to_mot.py: Convierte el ground-truth de ROBOT7 en formato COCO (labels.json,
# exportado de CVAT/FiftyOne) a formato MOTChallenge (gt.txt). La identidad de cada
# persona viene en annotation['attributes']['name'] (Jose, Miguel, Eugenio, ...).
#
# El nº de frame se asigna por ORDEN ALFABÉTICO del nombre de fichero de imagen
# (1..N), igual que hace track_boxmot.py al recorrer la carpeta, para que GT y
# predicción queden alineados.
#
# Uso:
#   python3 coco_to_mot.py --coco robot7/Seguimiento2Persona/COCO/labels.json \
#       --out robot7/Seguimiento2Persona/COCO/mot.txt

import argparse
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    d = json.load(open(args.coco))
    images = {im["id"]: im["file_name"] for im in d["images"]}
    # frame = índice (1..N) del fichero en orden alfabético
    order = {fn: i + 1 for i, fn in enumerate(sorted(images.values()))}

    # identidad: nombre -> entero estable (orden alfabético de nombres)
    names = sorted({a.get("attributes", {}).get("name", str(a.get("id")))
                    for a in d["annotations"]})
    name_id = {n: i + 1 for i, n in enumerate(names)}

    rows = []
    for a in d["annotations"]:
        fn = images[a["image_id"]]
        fr = order[fn]
        x, y, w, h = a["bbox"]
        nm = a.get("attributes", {}).get("name", str(a.get("id")))
        rows.append((fr, name_id[nm], x, y, w, h))
    rows.sort(key=lambda r: (r[0], r[1]))

    with open(args.out, "w") as f:
        for fr, i, x, y, w, h in rows:
            f.write(f"{fr},{i},{x:.0f},{y:.0f},{w:.0f},{h:.0f},1,-1,-1,-1\n")
    print(f"[SUCCESS] {len(rows)} cajas, {len(order)} frames, identidades {name_id} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

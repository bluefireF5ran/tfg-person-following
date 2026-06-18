#!/usr/bin/env python3
"""Verify GT bounding boxes by overlaying them on images.

Generates a mosaic of sample images with GT boxes drawn, for visual
inspection of the automatic labeling quality.

Usage:
  python3 verify_gt.py --data_dir ../data/sim_dataset --n 25
"""
import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


def parse_voc_xml(xml_path):
    """Read PASCAL VOC XML, return list of (label, xmin, ymin, xmax, ymax)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bb = obj.find("bndbox")
        x1 = int(bb.find("xmin").text)
        y1 = int(bb.find("ymin").text)
        x2 = int(bb.find("xmax").text)
        y2 = int(bb.find("ymax").text)
        bboxes.append((label, x1, y1, x2, y2))
    return bboxes


def main():
    parser = argparse.ArgumentParser(description="Verify GT annotations visually")
    parser.add_argument("--data_dir", required=True, help="Dataset directory")
    parser.add_argument("--n", type=int, default=25, help="Number of samples (default 25)")
    parser.add_argument("--output", type=str, default="gt_verification.jpg",
                        help="Output mosaic image")
    args = parser.parse_args()

    img_dir = Path(args.data_dir) / "images"
    lbl_dir = Path(args.data_dir) / "labels"

    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        print(f"No images found in {img_dir}")
        return

    n = min(args.n, len(images))
    samples = random.sample(images, n) if len(images) > n else images[:n]

    # Determine grid layout
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    # Read first image for tile size
    sample_img = cv2.imread(str(samples[0]))
    th, tw = sample_img.shape[:2]

    mosaic = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)

    total_boxes = 0
    imgs_with_person = 0

    for i, img_path in enumerate(samples):
        row, col = divmod(i, cols)
        img = cv2.imread(str(img_path))

        xml_path = lbl_dir / f"{img_path.stem}.xml"
        bboxes = parse_voc_xml(xml_path) if xml_path.exists() else []

        if bboxes:
            imgs_with_person += 1
        total_boxes += len(bboxes)

        for label, x1, y1, x2, y2 in bboxes:
            color = (0, 255, 0) if label == "person" else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Image name overlay
        cv2.putText(img, img_path.name, (2, th - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        mosaic[row * th:(row + 1) * th, col * tw:(col + 1) * tw] = img

    out_path = Path(args.data_dir) / args.output
    cv2.imwrite(str(out_path), mosaic)

    print(f"Mosaic saved: {out_path}")
    print(f"  Samples:        {n}")
    print(f"  With person GT: {imgs_with_person}/{n}")
    print(f"  Total GT boxes: {total_boxes}")
    print(f"  Avg boxes/img:  {total_boxes / n:.1f}")


if __name__ == "__main__":
    main()

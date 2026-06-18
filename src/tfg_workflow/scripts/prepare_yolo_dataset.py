#!/usr/bin/env python3
"""Prepare YOLO training dataset from PASCAL VOC annotations.

Converts PASCAL VOC XML labels to YOLO .txt format and creates
train/val/test split with data.yaml for Ultralytics training.

Supports two modes:
  1. Single dataset (sim_dataset): converts existing labels
  2. Combined dataset: merges sim_dataset + extended captures

Usage:
    # Basic: convert existing sim_dataset
    python prepare_yolo_dataset.py

    # Specify paths
    python prepare_yolo_dataset.py \
        --images_dir ../data/sim_dataset/images \
        --labels_dir ../data/sim_dataset/labels \
        --output_dir ../data/yolo_dataset \
        --split 0.8 0.1 0.1
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def voc_to_yolo(xml_path: Path, img_w: int = None, img_h: int = None) -> list[str]:
    """Convert PASCAL VOC XML to YOLO format lines.

    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1].

    Returns list of YOLO annotation strings (one per object).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image dimensions from XML if not provided
    size = root.find("size")
    if size is not None:
        w = int(size.find("width").text)
        h = int(size.find("height").text)
    else:
        w, h = img_w, img_h

    if w is None or h is None:
        raise ValueError(f"Cannot determine image size for {xml_path}")

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.lower()
        # Map class names to IDs
        # For our case, only "person" → class 0
        class_map = {"person": 0}
        if name not in class_map:
            continue
        class_id = class_map[name]

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # Convert to YOLO format (normalized center + width/height)
        x_center = ((xmin + xmax) / 2.0) / w
        y_center = ((ymin + ymax) / 2.0) / h
        bbox_w = (xmax - xmin) / w
        bbox_h = (ymax - ymin) / h

        # Clamp to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        bbox_w = max(0.0, min(1.0, bbox_w))
        bbox_h = max(0.0, min(1.0, bbox_h))

        if bbox_w > 0.001 and bbox_h > 0.001:
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}")

    return lines


def prepare_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    copy_images: bool = True,
):
    """Convert VOC dataset to YOLO format with train/val/test split.

    Creates:
        output_dir/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── labels/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── data.yaml
    """
    # Find all images with corresponding annotations
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in extensions
    )

    # Filter to images that have annotations
    paired = []
    empty = 0
    for img_path in image_files:
        xml_path = labels_dir / f"{img_path.stem}.xml"
        if xml_path.exists():
            # Check if annotation has objects
            lines = voc_to_yolo(xml_path)
            if lines:
                paired.append((img_path, xml_path, lines))
            else:
                empty += 1
        # else: no annotation → skip (could be background images)

    print(f"Found {len(paired)} images with annotations ({empty} empty annotations skipped)")
    print(f"Total objects: {sum(len(lines) for _, _, lines in paired)}")

    if not paired:
        print("ERROR: No paired images found!")
        return

    # Shuffle with fixed seed for reproducibility
    import random
    rng = random.Random(seed)
    rng.shuffle(paired)

    # Split
    n = len(paired)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # rest goes to test
    n_test = n - n_train - n_val

    splits = {
        "train": paired[:n_train],
        "val": paired[n_train : n_train + n_val],
        "test": paired[n_train + n_val :],
    }

    print(f"\nSplit: train={n_train}, val={n_val}, test={n_test}")

    # Create output directories
    for split_name in splits:
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    # Copy/symlink files
    total_objs = 0
    for split_name, items in splits.items():
        split_objs = 0
        for img_path, xml_path, yolo_lines in items:
            # Copy image
            dst_img = output_dir / "images" / split_name / img_path.name
            if copy_images:
                shutil.copy2(img_path, dst_img)
            else:
                dst_img.symlink_to(img_path.resolve())

            # Write YOLO label
            dst_label = output_dir / "labels" / split_name / f"{img_path.stem}.txt"
            dst_label.write_text("\n".join(yolo_lines) + "\n")
            split_objs += len(yolo_lines)

        total_objs += split_objs
        print(f"  {split_name}: {len(items)} images, {split_objs} objects")

    # Generate data.yaml
    data_yaml = output_dir / "data.yaml"
    yaml_content = f"""# YOLO training dataset — Person detection in Gazebo simulation
# Generated by prepare_yolo_dataset.py

path: {output_dir.resolve()}
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: person
"""
    data_yaml.write_text(yaml_content)
    print(f"\n✓ data.yaml written to {data_yaml}")
    print(f"  path: {output_dir.resolve()}")
    print(f"  Total: {len(paired)} images, {total_objs} annotations")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PASCAL VOC dataset to YOLO training format"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Input images directory (default: ../data/sim_dataset/images)",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default=None,
        help="Input VOC labels directory (default: ../data/sim_dataset/labels)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output YOLO dataset directory (default: ../data/yolo_dataset)",
    )
    parser.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        help="Train/val/test split ratios (default: 0.8 0.1 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks instead of copying images",
    )
    args = parser.parse_args()

    # Default paths relative to script location
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent / "data"

    images_dir = Path(args.images_dir) if args.images_dir else base_dir / "sim_dataset" / "images"
    labels_dir = Path(args.labels_dir) if args.labels_dir else base_dir / "sim_dataset" / "labels"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "yolo_dataset"

    print("=" * 60)
    print("YOLO Dataset Preparation")
    print("=" * 60)
    print(f"Images:   {images_dir}")
    print(f"Labels:   {labels_dir}")
    print(f"Output:   {output_dir}")
    print(f"Split:    {args.split[0]}/{args.split[1]}/{args.split[2]}")
    print(f"Seed:     {args.seed}")
    print()

    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        return
    if not labels_dir.exists():
        print(f"ERROR: Labels directory not found: {labels_dir}")
        return

    prepare_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
        train_ratio=args.split[0],
        val_ratio=args.split[1],
        test_ratio=args.split[2],
        seed=args.seed,
        copy_images=not args.symlink,
    )

    print("\nDone! Ready for training with:")
    print(f"  yolo train model=yolo11s.pt data={output_dir}/data.yaml imgsz=240")
    print(f"  python scripts/train_yolo.py --data {output_dir}/data.yaml")


if __name__ == "__main__":
    main()
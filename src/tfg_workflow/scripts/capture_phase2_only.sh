#!/usr/bin/env bash
# ===========================================================
# capture_phase2_only.sh — Capture Phase 2 only (Varied world)
# ===========================================================
# This script continues capture from Phase 1 completion
# Start index: 5940 (after Phase 1 completion)
# Target: ~5940 additional images from varied world
# ===========================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLD_DIR="$SCRIPT_DIR/../worlds"
DATA_DIR="$SCRIPT_DIR/../data/sim_dataset_extended"
SDF_VARIED="$WORLD_DIR/warehouse_actors_varied.sdf"
CAPTURE_SCRIPT="$SCRIPT_DIR/capture_with_gt.py"

RATE=3.0          # Hz
SETTLE=1.5        # seconds
IMAGES_PER_WP=30  # images per waypoint

# Start from where Phase 1 ended
START_INDEX=5940

echo "============================================"
echo "  Phase 2 Only - Varied World Capture"
echo "============================================"
echo "SDF varied: $SDF_VARIED"
echo "Output:     $DATA_DIR"
echo "Start index: $START_INDEX"
echo "Rate:       ${RATE} Hz"
echo "Images/WP:  $IMAGES_PER_WP"
echo ""

# Source ROS 2
set +u
source /opt/ros/jazzy/setup.bash 2>/dev/null || true
source ~/tfg_ws/install/setup.bash 2>/dev/null || true
set -u

# ── Same waypoint batches as Phase 2 of original script ────────
# Phase 2 batches (26-50): varied world with same positions

WP_B26="-5.5,1.5,0 -5.5,1.5,45 -5.5,1.5,315 -4,1.5,0 -4,1.5,45 -4,1.5,90 -2.5,1.5,0 -2.5,1.5,45 -2.5,1.5,315"
WP_B27="-1,1.5,0 -1,1.5,45 -1,1.5,90 0.5,1.5,0 0.5,1.5,45 0.5,1.5,90 0.5,1.5,315 2,1.5,0 2,1.5,45"
WP_B28="3.5,1.5,0 3.5,1.5,315 5,1.5,0 5,1.5,315 5.5,1.5,0 5.5,1.5,180 5.5,1.5,270"

WP_B29="-5.5,2.5,0 -5.5,2.5,45 -5.5,2.5,90 -5.5,2.5,315 -4,2.5,0 -4,2.5,45 -4,2.5,90 -4,2.5,180"
WP_B30="-2.5,2.5,0 -2.5,2.5,45 -2.5,2.5,90 -2.5,2.5,180 -2.5,2.5,270 -1,2.5,0 -1,2.5,45 -1,2.5,90"
WP_B31="0.5,2.5,0 0.5,2.5,45 0.5,2.5,90 0.5,2.5,180 0.5,2.5,270 2,2.5,0 2,2.5,45 2,2.5,315"
WP_B32="3.5,2.5,0 3.5,2.5,90 3.5,2.5,180 3.5,2.5,270 5,2.5,0 5,2.5,135 5,2.5,180 5,2.5,225"

WP_B33="-5.5,3.5,0 -5.5,3.5,45 -5.5,3.5,90 -4,3.5,0 -4,3.5,45 -4,3.5,90 -4,3.5,135 -4,3.5,180"
WP_B34="-2.5,3.5,0 -2.5,3.5,45 -2.5,3.5,90 -2.5,3.5,180 -2.5,3.5,270 -2.5,3.5,315 -1,3.5,0 -1,3.5,90"
WP_B35="0.5,3.5,0 0.5,3.5,90 0.5,3.5,180 0.5,3.5,270 2,3.5,0 2,3.5,90 2,3.5,180 2,3.5,270"
WP_B36="3.5,3.5,0 3.5,3.5,90 3.5,3.5,180 3.5,3.5,270 5,3.5,0 5,3.5,90 5,3.5,180 5.5,3.5,180"

WP_B37="-5.5,4.5,90 -5.5,4.5,135 -5.5,4.5,180 -4,4.5,0 -4,4.5,90 -4,4.5,135 -4,4.5,180 -4,4.5,270"
WP_B38="-2.5,4.5,0 -2.5,4.5,45 -2.5,4.5,90 -2.5,4.5,180 -2.5,4.5,270 -1,4.5,0 -1,4.5,90 -1,4.5,180"
WP_B39="0.5,4.5,0 0.5,4.5,90 0.5,4.5,180 0.5,4.5,270 2,4.5,0 2,4.5,90 2,4.5,180 2,4.5,270"
WP_B40="3.5,4.5,0 3.5,4.5,90 3.5,4.5,180 3.5,4.5,270 5,4.5,0 5,4.5,135 5,4.5,180 5.5,4.5,135"

WP_B41="-5.5,5.5,90 -5.5,5.5,135 -5.5,5.5,180 -4,5.5,90 -4,5.5,135 -4,5.5,180 -4,5.5,225"
WP_B42="-2.5,5.5,0 -2.5,5.5,90 -2.5,5.5,180 -2.5,5.5,270 -1,5.5,0 -1,5.5,90 -1,5.5,180"
WP_B43="0.5,5.5,0 0.5,5.5,90 0.5,5.5,180 0.5,5.5,270 2,5.5,0 2,5.5,90 2,5.5,180 2,5.5,270"
WP_B44="3.5,5.5,0 3.5,5.5,90 3.5,5.5,180 3.5,5.5,270 5,5.5,90 5,5.5,135 5,5.5,180 5.5,5.5,180"

WP_B45="-5.5,6.0,90 -5.5,6.0,135 -5.5,6.0,180 -4,6.0,90 -4,6.0,135 -4,6.0,225 -2.5,6.0,90 -2.5,6.0,180 -2.5,6.0,270"
WP_B46="-1,6.0,90 -1,6.0,180 -1,6.0,270 0.5,6.0,90 0.5,6.0,180 0.5,6.0,270 2,6.0,90 2,6.0,180 2,6.0,270"
WP_B47="3.5,6.0,90 3.5,6.0,180 3.5,6.0,270 5,6.0,90 5,6.0,180 5,6.0,225 5.5,6.0,135 5.5,6.0,180 5.5,6.0,225"

WP_B48="-5.5,2,30 -5.5,3,60 -5.5,4,120 -5.5,5,150 5.5,2,330 5.5,3,300 5.5,4,240 5.5,5,210"
WP_B49="-5,1.5,15 -5,3,75 -5,5,105 5,1.5,345 5,3,285 5,5,255"
WP_B50="-3,1.5,0 -3,6.5,180 3,1.5,0 3,6.5,180 0,1.5,90 0,6.5,270"

ALL_BATCHES_VARIED=(
    "$WP_B26" "$WP_B27" "$WP_B28"
    "$WP_B29" "$WP_B30" "$WP_B31" "$WP_B32"
    "$WP_B33" "$WP_B34" "$WP_B35" "$WP_B36"
    "$WP_B37" "$WP_B38" "$WP_B39" "$WP_B40"
    "$WP_B41" "$WP_B42" "$WP_B43" "$WP_B44"
    "$WP_B45" "$WP_B46" "$WP_B47"
    "$WP_B48" "$WP_B49" "$WP_B50"
)

# Count total
total_wps=0
for batch in "${ALL_BATCHES_VARIED[@]}"; do
    n=$(echo "$batch" | wc -w)
    total_wps=$((total_wps + n))
done
total_images=$((total_wps * IMAGES_PER_WP))

echo "Phase 2 waypoints:    $total_wps"
echo "Images per waypoint:  $IMAGES_PER_WP"
echo "Estimated Phase 2:    $total_images images"
echo "Final total:          $((START_INDEX + total_images)) images"
echo ""

# ── Verify prerequisites ─────────────────────────────────────
if ! ros2 topic list 2>/dev/null | grep -q '/oakd/rgb/preview/image_raw'; then
    echo "ERROR: Camera topic not found. Is the simulation running?"
    echo "  Start with: bash scripts/run_pipeline.sh sim-varied"
    exit 1
fi
echo "✓ Camera topic active"
echo ""

# ── Run capture batches ──────────────────────────────────────
img_index=$START_INDEX
batch_num=0
total_batches=${#ALL_BATCHES_VARIED[@]}

run_batch() {
    local sdf="$1"
    local world_name="$2"
    local waypoints="$3"

    local n_wps
    n_wps=$(echo "$waypoints" | wc -w)
    local batch_imgs=$((n_wps * IMAGES_PER_WP))

    batch_num=$((batch_num + 1))
    echo ""
    echo "──────────────────────────────────────────"
    echo "  Batch $batch_num/$total_batches:  $n_wps wps, ~$batch_imgs imgs [$world_name]"
    echo "  Starting at index $img_index"
    echo "──────────────────────────────────────────"

    python3 "$CAPTURE_SCRIPT" \
        --sdf "$sdf" \
        --output_dir "$DATA_DIR" \
        --rate "$RATE" \
        --waypoints "$waypoints" \
        --images_per_wp "$IMAGES_PER_WP" \
        --settle_time "$SETTLE" \
        --start_index "$img_index" \
        --world "$world_name" \
        --ros-args -p use_sim_time:=true

    img_index=$((img_index + batch_imgs))
}

# ── Phase 2: Varied world (all batches) ───────────────────────
echo ""
echo "========= Phase 2: Varied World ============="
echo "Capturing ${#ALL_BATCHES_VARIED[@]} batches from warehouse_actors_varied.sdf"
echo ""

for batch in "${ALL_BATCHES_VARIED[@]}"; do
    run_batch "$SDF_VARIED" "warehouse_actors_varied" "$batch"
done

echo ""
echo "============================================"
echo "  Phase 2 Capture Complete!"
echo "============================================"
n_imgs=$(ls "$DATA_DIR/images/" 2>/dev/null | wc -l)
n_xmls=$(ls "$DATA_DIR/labels/" 2>/dev/null | wc -l)
echo "  Total images:      $n_imgs"
echo "  Total labels (XML): $n_xmls"
echo "  Phase 2 added:     $((n_imgs - START_INDEX))"
echo "  Output:            $DATA_DIR"
echo ""
echo "Next steps:"
echo "  1. Verify GT:    python3 scripts/verify_gt.py --data_dir $DATA_DIR"
echo "  2. Prepare YOLO: python3 scripts/prepare_yolo_dataset.py --images_dir $DATA_DIR/images --labels_dir $DATA_DIR/labels --output_dir ../data/yolo_dataset_extended"
echo "  3. Train:        python3 scripts/train_yolo.py --data ../data/yolo_dataset_extended/data.yaml --model yolo11s.pt"

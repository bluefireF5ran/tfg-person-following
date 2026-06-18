#!/usr/bin/env bash
# ===========================================================
# capture_dataset_extended.sh — Capture ~8000+ images for training
# ===========================================================
#
# Extended version of capture_dataset.sh that generates a much larger
# dataset suitable for YOLO fine-tuning. The reference TFG (Cano Sampedro)
# used ~8000 images. This script captures:
#   - ~4000 images from warehouse_actors.sdf (walk animations)
#   - ~4000 images from warehouse_actors_varied.sdf (mixed animations)
#
# Strategy for variety:
#   - Dense waypoint grid covering the entire safe zone
#   - Multiple yaw orientations per position (8 directions)
#   - Higher images_per_wp to capture actor movement over time
#   - Both world variants for animation diversity
#
# Prerequisites:
#   - Simulation running with appropriate world
#
# Usage:
#   cd ~/tfg_ws/src/tfg_workflow
#   bash scripts/capture_dataset_extended.sh
# ===========================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLD_DIR="$SCRIPT_DIR/../worlds"
DATA_DIR="$SCRIPT_DIR/../data/sim_dataset_extended"
SDF_BASE="$WORLD_DIR/warehouse_actors.sdf"
SDF_VARIED="$WORLD_DIR/warehouse_actors_varied.sdf"
CAPTURE_SCRIPT="$SCRIPT_DIR/capture_with_gt.py"

RATE=3.0          # Hz — faster capture rate
SETTLE=1.5        # shorter settle (1.5s enough for camera)
IMAGES_PER_WP=30  # more images per waypoint (captures actor movement)

echo "============================================"
echo "  Extended Dataset Capture (~8000 images)"
echo "============================================"
echo "SDF base:   $SDF_BASE"
echo "SDF varied: $SDF_VARIED"
echo "Output:     $DATA_DIR"
echo "Rate:       ${RATE} Hz"
echo "Images/WP:  $IMAGES_PER_WP"
echo ""

# Source ROS 2
set +u
source /opt/ros/jazzy/setup.bash 2>/dev/null || true
source ~/tfg_ws/install/setup.bash 2>/dev/null || true
set -u

# ── Dense waypoint grid ──────────────────────────────────────
# Safe zone: X ∈ [-6, 6], Y ∈ [1, 6.5]
# Grid: X every 1.5m, Y every 1m → ~80 positions
# 8 yaw angles per position → ~640 waypoints
# At 30 imgs/wp → ~19,200 images (we'll use a subset)

# ── Grid positions (systematic coverage) ─────────────────────
# X positions: -5.5, -4, -2.5, -1, 0.5, 2, 3.5, 5, 5.5
# Y positions: 1.5, 2.5, 3.5, 4.5, 5.5, 6.0
# Yaw angles: 0, 45, 90, 135, 180, 225, 270, 315

# ── Batch 1: Southern corridor (Y=1.5) — close to actor_0,5 ─
WP_B01="-5.5,1.5,0 -5.5,1.5,45 -5.5,1.5,315 -4,1.5,0 -4,1.5,45 -4,1.5,90 -2.5,1.5,0 -2.5,1.5,45 -2.5,1.5,315"
WP_B02="-1,1.5,0 -1,1.5,45 -1,1.5,90 0.5,1.5,0 0.5,1.5,45 0.5,1.5,90 0.5,1.5,315 2,1.5,0 2,1.5,45"
WP_B03="3.5,1.5,0 3.5,1.5,315 5,1.5,0 5,1.5,315 5.5,1.5,0 5.5,1.5,180 5.5,1.5,270"

# ── Batch 2: Y=2.5 — central room south ─────────────────────
WP_B04="-5.5,2.5,0 -5.5,2.5,45 -5.5,2.5,90 -5.5,2.5,315 -4,2.5,0 -4,2.5,45 -4,2.5,90 -4,2.5,180"
WP_B05="-2.5,2.5,0 -2.5,2.5,45 -2.5,2.5,90 -2.5,2.5,180 -2.5,2.5,270 -1,2.5,0 -1,2.5,45 -1,2.5,90"
WP_B06="0.5,2.5,0 0.5,2.5,45 0.5,2.5,90 0.5,2.5,180 0.5,2.5,270 2,2.5,0 2,2.5,45 2,2.5,315"
WP_B07="3.5,2.5,0 3.5,2.5,90 3.5,2.5,180 3.5,2.5,270 5,2.5,0 5,2.5,135 5,2.5,180 5,2.5,225"

# ── Batch 3: Y=3.5 — central room middle ────────────────────
WP_B08="-5.5,3.5,0 -5.5,3.5,45 -5.5,3.5,90 -4,3.5,0 -4,3.5,45 -4,3.5,90 -4,3.5,135 -4,3.5,180"
WP_B09="-2.5,3.5,0 -2.5,3.5,45 -2.5,3.5,90 -2.5,3.5,180 -2.5,3.5,270 -2.5,3.5,315 -1,3.5,0 -1,3.5,90"
WP_B10="0.5,3.5,0 0.5,3.5,90 0.5,3.5,180 0.5,3.5,270 2,3.5,0 2,3.5,90 2,3.5,180 2,3.5,270"
WP_B11="3.5,3.5,0 3.5,3.5,90 3.5,3.5,180 3.5,3.5,270 5,3.5,0 5,3.5,90 5,3.5,180 5.5,3.5,180"

# ── Batch 4: Y=4.5 — central room north ─────────────────────
WP_B12="-5.5,4.5,90 -5.5,4.5,135 -5.5,4.5,180 -4,4.5,0 -4,4.5,90 -4,4.5,135 -4,4.5,180 -4,4.5,270"
WP_B13="-2.5,4.5,0 -2.5,4.5,45 -2.5,4.5,90 -2.5,4.5,180 -2.5,4.5,270 -1,4.5,0 -1,4.5,90 -1,4.5,180"
WP_B14="0.5,4.5,0 0.5,4.5,90 0.5,4.5,180 0.5,4.5,270 2,4.5,0 2,4.5,90 2,4.5,180 2,4.5,270"
WP_B15="3.5,4.5,0 3.5,4.5,90 3.5,4.5,180 3.5,4.5,270 5,4.5,0 5,4.5,135 5,4.5,180 5.5,4.5,135"

# ── Batch 5: Y=5.5 — north aisle ────────────────────────────
WP_B16="-5.5,5.5,90 -5.5,5.5,135 -5.5,5.5,180 -4,5.5,90 -4,5.5,135 -4,5.5,180 -4,5.5,225"
WP_B17="-2.5,5.5,0 -2.5,5.5,90 -2.5,5.5,180 -2.5,5.5,270 -1,5.5,0 -1,5.5,90 -1,5.5,180"
WP_B18="0.5,5.5,0 0.5,5.5,90 0.5,5.5,180 0.5,5.5,270 2,5.5,0 2,5.5,90 2,5.5,180 2,5.5,270"
WP_B19="3.5,5.5,0 3.5,5.5,90 3.5,5.5,180 3.5,5.5,270 5,5.5,90 5,5.5,135 5,5.5,180 5.5,5.5,180"

# ── Batch 6: Y=6.0 — top aisle ──────────────────────────────
WP_B20="-5.5,6.0,90 -5.5,6.0,135 -5.5,6.0,180 -4,6.0,90 -4,6.0,135 -4,6.0,225 -2.5,6.0,90 -2.5,6.0,180 -2.5,6.0,270"
WP_B21="-1,6.0,90 -1,6.0,180 -1,6.0,270 0.5,6.0,90 0.5,6.0,180 0.5,6.0,270 2,6.0,90 2,6.0,180 2,6.0,270"
WP_B22="3.5,6.0,90 3.5,6.0,180 3.5,6.0,270 5,6.0,90 5,6.0,180 5,6.0,225 5.5,6.0,135 5.5,6.0,180 5.5,6.0,225"

# ── Batch 7: Diagonal / corner positions ─────────────────────
WP_B23="-5.5,2,30 -5.5,3,60 -5.5,4,120 -5.5,5,150 5.5,2,330 5.5,3,300 5.5,4,240 5.5,5,210"
WP_B24="-5,1.5,15 -5,3,75 -5,5,105 5,1.5,345 5,3,285 5,5,255"
WP_B25="-3,1.5,0 -3,6.5,180 3,1.5,0 3,6.5,180 0,1.5,90 0,6.5,270"

ALL_BATCHES_BASE=(
    "$WP_B01" "$WP_B02" "$WP_B03"
    "$WP_B04" "$WP_B05" "$WP_B06" "$WP_B07"
    "$WP_B08" "$WP_B09" "$WP_B10" "$WP_B11"
    "$WP_B12" "$WP_B13" "$WP_B14" "$WP_B15"
    "$WP_B16" "$WP_B17" "$WP_B18" "$WP_B19"
    "$WP_B20" "$WP_B21" "$WP_B22"
    "$WP_B23" "$WP_B24" "$WP_B25"
)

# Varied world uses same positions but different actor animations
ALL_BATCHES_VARIED=(
    "$WP_B01" "$WP_B02" "$WP_B03"
    "$WP_B04" "$WP_B05" "$WP_B06" "$WP_B07"
    "$WP_B08" "$WP_B09" "$WP_B10" "$WP_B11"
    "$WP_B12" "$WP_B13" "$WP_B14" "$WP_B15"
    "$WP_B16" "$WP_B17" "$WP_B18" "$WP_B19"
    "$WP_B20" "$WP_B21" "$WP_B22"
    "$WP_B23" "$WP_B24" "$WP_B25"
)

# Count total
total_wps=0
for batch in "${ALL_BATCHES_BASE[@]}"; do
    n=$(echo "$batch" | wc -w)
    total_wps=$((total_wps + n))
done
for batch in "${ALL_BATCHES_VARIED[@]}"; do
    n=$(echo "$batch" | wc -w)
    total_wps=$((total_wps + n))
done
total_images=$((total_wps * IMAGES_PER_WP))

echo "Total waypoints:       $total_wps"
echo "Images per waypoint:   $IMAGES_PER_WP"
echo "Estimated total imgs:  $total_images"
echo ""

# ── Verify prerequisites ─────────────────────────────────────
if ! ros2 topic list 2>/dev/null | grep -q '/oakd/rgb/preview/image_raw'; then
    echo "ERROR: Camera topic not found. Is the simulation running?"
    echo "  Start with: bash scripts/run_pipeline.sh sim"
    exit 1
fi
echo "✓ Camera topic active"

# ── Run capture batches ──────────────────────────────────────
img_index=0
batch_num=0
total_batches=$(( ${#ALL_BATCHES_BASE[@]} + ${#ALL_BATCHES_VARIED[@]} ))

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

# Phase 1: Base world (all actors walking)
echo ""
echo "============ Phase 1: Base world ============"
echo "Capturing ${#ALL_BATCHES_BASE[@]} batches from warehouse_actors.sdf"
echo ""
for batch in "${ALL_BATCHES_BASE[@]}"; do
    run_batch "$SDF_BASE" "warehouse_actors" "$batch"
done

# Phase 2: Varied world
echo ""
echo "========= Phase 2: Varied world ============="
echo "The simulation must be restarted with the varied world."
echo "  bash scripts/run_pipeline.sh sim-varied"
echo ""
read -r -p "Press ENTER when the varied simulation is ready (or Ctrl-C to stop)..."

for batch in "${ALL_BATCHES_VARIED[@]}"; do
    run_batch "$SDF_VARIED" "warehouse_actors_varied" "$batch"
done

echo ""
echo "============================================"
echo "  Extended Capture Complete!"
echo "============================================"
n_imgs=$(ls "$DATA_DIR/images/" 2>/dev/null | wc -l)
n_xmls=$(ls "$DATA_DIR/labels/" 2>/dev/null | wc -l)
echo "  Images:      $n_imgs"
echo "  Labels (XML): $n_xmls"
echo "  Output:       $DATA_DIR"
echo ""
echo "Next steps:"
echo "  1. Verify GT:    python3 scripts/verify_gt.py --data_dir $DATA_DIR"
echo "  2. Prepare YOLO: python3 scripts/prepare_yolo_dataset.py --images_dir $DATA_DIR/images --labels_dir $DATA_DIR/labels --output_dir ../data/yolo_dataset_extended"
echo "  3. Train:        python3 scripts/train_yolo.py --data ../data/yolo_dataset_extended/data.yaml --model yolo11s.pt"
</task_progress>
</write_to_file>
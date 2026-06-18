#!/usr/bin/env bash
# ===========================================================
# capture_phase2_resume.sh — Resume Phase 2 from batch 9
# ===========================================================
# Continues capture from where it stopped: batch 9/25
# Start index: 8095 (after 8095 images captured)
# Target: ~3785 additional images to complete dataset
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

# Continue from where we stopped
START_INDEX=8095

echo "============================================"
echo "  Phase 2 Resume - From Batch 9"
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

# ── Batches 9-25: Continue from batch 9 (where we stopped) ─────

# Batch 9 (8 waypoints)
WP_B09="-2.5,3.5,0 -2.5,3.5,45 -2.5,3.5,90 -2.5,3.5,180 -2.5,3.5,270 -2.5,3.5,315 -1,3.5,0 -1,3.5,90"

# Batch 10 (8 waypoints)
WP_B10="0.5,3.5,0 0.5,3.5,90 0.5,3.5,180 0.5,3.5,270 2,3.5,0 2,3.5,90 2,3.5,180 2,3.5,270"

# Batch 11 (8 waypoints)
WP_B11="3.5,3.5,0 3.5,3.5,90 3.5,3.5,180 3.5,3.5,270 5,3.5,0 5,3.5,90 5,3.5,180 5.5,3.5,180"

# Batch 12 (8 waypoints)
WP_B12="-5.5,4.5,90 -5.5,4.5,135 -5.5,4.5,180 -4,4.5,0 -4,4.5,90 -4,4.5,135 -4,4.5,180 -4,4.5,270"

# Batch 13 (8 waypoints)
WP_B13="-2.5,4.5,0 -2.5,4.5,45 -2.5,4.5,90 -2.5,4.5,180 -2.5,4.5,270 -1,4.5,0 -1,4.5,90 -1,4.5,180"

# Batch 14 (8 waypoints)
WP_B14="0.5,4.5,0 0.5,4.5,90 0.5,4.5,180 0.5,4.5,270 2,4.5,0 2,4.5,90 2,4.5,180 2,4.5,270"

# Batch 15 (8 waypoints)
WP_B15="3.5,4.5,0 3.5,4.5,90 3.5,4.5,180 3.5,4.5,270 5,4.5,0 5,4.5,135 5,4.5,180 5.5,4.5,135"

# Batch 16 (7 waypoints)
WP_B16="-5.5,5.5,90 -5.5,5.5,135 -5.5,5.5,180 -4,5.5,90 -4,5.5,135 -4,5.5,180 -4,5.5,225"

# Batch 17 (7 waypoints)
WP_B17="-2.5,5.5,0 -2.5,5.5,90 -2.5,5.5,180 -2.5,5.5,270 -1,5.5,0 -1,5.5,90 -1,5.5,180"

# Batch 18 (8 waypoints)
WP_B18="0.5,5.5,0 0.5,5.5,90 0.5,5.5,180 0.5,5.5,270 2,5.5,0 2,5.5,90 2,5.5,180 2,5.5,270"

# Batch 19 (8 waypoints)
WP_B19="3.5,5.5,0 3.5,5.5,90 3.5,5.5,180 3.5,5.5,270 5,5.5,90 5,5.5,135 5,5.5,180 5.5,5.5,180"

# Batch 20 (9 waypoints)
WP_B20="-5.5,6.0,90 -5.5,6.0,135 -5.5,6.0,180 -4,6.0,90 -4,6.0,135 -4,6.0,225 -2.5,6.0,90 -2.5,6.0,180 -2.5,6.0,270"

# Batch 21 (9 waypoints)
WP_B21="-1,6.0,90 -1,6.0,180 -1,6.0,270 0.5,6.0,90 0.5,6.0,180 0.5,6.0,270 2,6.0,90 2,6.0,180 2,6.0,270"

# Batch 22 (8 waypoints)
WP_B22="3.5,6.0,90 3.5,6.0,180 3.5,6.0,270 5,6.0,90 5,6.0,180 5,6.0,225 5.5,6.0,135 5.5,6.0,180 5.5,6.0,225"

# Batch 23 (8 waypoints)
WP_B23="-5.5,2,30 -5.5,3,60 -5.5,4,120 -5.5,5,150 5.5,2,330 5.5,3,300 5.5,4,240 5.5,5,210"

# Batch 24 (6 waypoints)
WP_B24="-5,1.5,15 -5,3,75 -5,5,105 5,1.5,345 5,3,285 5,5,255"

# Batch 25 (6 waypoints)
WP_B25="-3,1.5,0 -3,6.5,180 3,1.5,0 3,6.5,180 0,1.5,90 0,6.5,270"

REMAINING_BATCHES=(
    "$WP_B09" "$WP_B10" "$WP_B11" "$WP_B12" "$WP_B13"
    "$WP_B14" "$WP_B15" "$WP_B16" "$WP_B17" "$WP_B18"
    "$WP_B19" "$WP_B20" "$WP_B21" "$WP_B22" "$WP_B23"
    "$WP_B24" "$WP_B25"
)

# Count remaining
total_wps=0
for batch in "${REMAINING_BATCHES[@]}"; do
    n=$(echo "$batch" | wc -w)
    total_wps=$((total_wps + n))
done
total_images=$((total_wps * IMAGES_PER_WP))

echo "Remaining batches:    ${#REMAINING_BATCHES[@]}"
echo "Remaining waypoints:  $total_wps"
echo "Estimated remaining:  $total_images images"
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
batch_num=8  # We're continuing from batch 9 (0-indexed as 8)
total_batches=${#REMAINING_BATCHES[@]}

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
    echo "  Batch $batch_num/25:  $n_wps wps, ~$batch_imgs imgs [$world_name]"
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

# ── Phase 2 Resume: batches 9-25 (varied world) ─────────────────
echo ""
echo "========= Phase 2 Resume: Batches 9-25 ============="
echo "Capturing ${#REMAINING_BATCHES[@]} remaining batches from warehouse_actors_varied.sdf"
echo ""

for batch in "${REMAINING_BATCHES[@]}"; do
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
echo "  Phase 2 added:     $((n_imgs - 5940))"
echo "  Output:            $DATA_DIR"
echo ""
echo "✅ Dataset ampliado completado!"
echo ""
echo "Next steps:"
echo "  1. Verify GT:    python3 scripts/verify_gt.py --data_dir $DATA_DIR"
echo "  2. Prepare YOLO: python3 scripts/prepare_yolo_dataset.py --images_dir $DATA_DIR/images --labels_dir $DATA_DIR/labels --output_dir ../data/yolo_dataset_extended"
echo "  3. Train:        python3 scripts/train_yolo.py --data ../data/yolo_dataset_extended/data.yaml --model yolo11s.pt"

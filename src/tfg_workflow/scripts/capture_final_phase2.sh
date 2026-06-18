#!/usr/bin/env bash
# ===========================================================
# capture_final_phase2.sh — FINAL ATTEMPT to complete dataset
# ===========================================================
# Continues from current status: 9163 images captured
# Target: 2717 remaining images to reach 11880 total
# If Gazebo fails again, we accept 9163 as final dataset
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
START_INDEX=9163
TARGET_TOTAL=11880
REMAINING=$((TARGET_TOTAL - START_INDEX))

echo "============================================"
echo "  FINAL ATTEMPT - Complete Dataset"
echo "============================================"
echo "SDF varied: $SDF_VARIED"
echo "Output:     $DATA_DIR"
echo "Start index: $START_INDEX"
echo "Target total: $TARGET_TOTAL"
echo "Remaining:   $REMAINING images"
echo ""
echo "⚠️  If Gazebo fails again, we accept $START_INDEX images"
echo "============================================"
echo ""

# Source ROS 2
set +u
source /opt/ros/jazzy/setup.bash 2>/dev/null || true
source ~/tfg_ws/install/setup.bash 2>/dev/null || true
set -u

# ── Verify prerequisites ─────────────────────────────────────
if ! ros2 topic list 2>/dev/null | grep -q '/oakd/rgb/preview/image_raw'; then
    echo "ERROR: Camera topic not found. Is the simulation running?"
    exit 1
fi
echo "✓ Camera topic active"
echo ""

# ── Remaining batches (14-25) ──────────────────────────────────────

# Batch 13 (8 waypoints) - complete this batch first
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
    "$WP_B13" "$WP_B14" "$WP_B15" "$WP_B16" "$WP_B17"
    "$WP_B18" "$WP_B19" "$WP_B20" "$WP_B21" "$WP_B22"
    "$WP_B23" "$WP_B24" "$WP_B25"
)

# ── Run capture batches ──────────────────────────────────────
img_index=$START_INDEX
batch_num=12  # We're continuing from batch 13

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
    echo "  Remaining: $((TARGET_TOTAL - img_index)) images"
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

    # Check if we've reached the target
    if [ $img_index -ge $TARGET_TOTAL ]; then
        echo ""
        echo "🎉 TARGET ALCANZADO: $img_index imágenes"
        return 0
    fi
}

# ── Capture remaining batches ───────────────────────────────
echo ""
echo "========= Capturing ${#REMAINING_BATCHES[@]} batches ============="
echo ""

for batch in "${REMAINING_BATCHES[@]}"; do
    run_batch "$SDF_VARIED" "warehouse_actors_varied" "$batch"

    # Check if we've reached target
    if [ $img_index -ge $TARGET_TOTAL ]; then
        break
    fi
done

echo ""
echo "============================================"
echo "  Capture Session Complete!"
echo "============================================"
n_imgs=$(ls "$DATA_DIR/images/" 2>/dev/null | wc -l)
n_xmls=$(ls "$DATA_DIR/labels/" 2>/dev/null | wc -l)
echo "  Total images:      $n_imgs"
echo "  Total labels (XML): $n_xmls"
echo "  Phase 1:          5940 (base world)"
echo "  Phase 2:          $((n_imgs - 5940)) (varied world)"
echo "  Output:           $DATA_DIR"
echo ""

if [ $n_imgs -ge $TARGET_TOTAL ]; then
    echo "✅ OBJETIVO COMPLETADO: Dataset ampliado listo!"
else
    echo "⚠️  Dataset parcial aceptado: $n_imgs imágenes"
fi

echo ""
echo "Next steps:"
echo "  1. Verify GT:    python3 scripts/verify_gt.py --data_dir $DATA_DIR"
echo "  2. Prepare YOLO: python3 scripts/prepare_yolo_dataset.py --images_dir $DATA_DIR/images --labels_dir $DATA_DIR/labels --output_dir ../data/yolo_dataset_extended"
echo "  3. Train:        python3 scripts/train_yolo.py --data ../data/yolo_dataset_extended/data.yaml --model yolo11s.pt"

#!/usr/bin/env bash
# ===========================================================
# capture_dataset.sh — Orchestrate diverse dataset capture
# ===========================================================
#
# Captures ~1600 images from the TurtleBot4 camera in Gazebo with
# automatic GT labels, using multiple robot positions & orientations
# to ensure dataset variety (distances, angles, backgrounds, occlusions).
#
# Prerequisites:
#   - Simulation running:  bash scripts/run_pipeline.sh sim
#   - Wait ~30s for camera topic to be active
#
# Usage:
#   cd ~/tfg_ws/src/tfg_workflow
#   bash scripts/capture_dataset.sh
#
# The script captures in batches, teleporting the robot between waypoints.
# Safe zone: X ∈ [-6, 6], Y ∈ [1, 6.5] (central warehouse room).
# ===========================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORLD_DIR="$SCRIPT_DIR/../worlds"
DATA_DIR="$SCRIPT_DIR/../data/sim_dataset"
SDF_BASE="$WORLD_DIR/warehouse_actors.sdf"
SDF_VARIED="$WORLD_DIR/warehouse_actors_varied.sdf"
CAPTURE_SCRIPT="$SCRIPT_DIR/capture_with_gt.py"

RATE=2.0          # Hz
SETTLE=2.0        # seconds to wait after teleport
IMAGES_PER_WP=20  # images per waypoint

echo "============================================"
echo "  Dataset Capture Orchestrator"
echo "============================================"
echo "SDF base:   $SDF_BASE"
echo "SDF varied: $SDF_VARIED"
echo "Output:     $DATA_DIR"
echo "Rate:       ${RATE} Hz"
echo "Settle:     ${SETTLE}s per waypoint"
echo ""

# Source ROS 2 (disable nounset — setup scripts use unbound vars)
set +u
source /opt/ros/jazzy/setup.bash 2>/dev/null || true
source ~/tfg_ws/install/setup.bash 2>/dev/null || true
set -u

# ── Waypoint definitions ──────────────────────────────────────
# Format: "x,y,yaw_deg" (yaw=0 faces East/+X, 90=North/+Y)
#
# Design criteria:
#   - Cover the safe zone systematically
#   - Mix distances to actors: close (1-2m), medium (3-4m), far (5-6m)
#   - Vary heading to get different backgrounds (shelves, walls, open)
#   - Include positions where actors cross camera view → occlusions
#   - Include positions where actors are partially behind obstacles
#
# Actor positions (approximate, they walk back and forth):
#   actor_0: Y≈2,   X: 0→5     (primary target)
#   actor_1: Y≈5.5, X: -4→3    (north aisle)
#   actor_2: X≈-4.5, Y: 1.5→6  (west corridor)
#   actor_3: Y≈4,   X: 2→5.5   (mid-north)
#   actor_4: X≈5.5, Y: 2→6     (east side)
#   actor_5: Y≈1.5, X: -5→-1   (south-west)
#   actor_6: Y≈6,   X: -2→4    (top aisle)
#   actor_7: X≈-1,  Y: 3→6     (center corridor)

# ── Batch 1: Close range following actor_0 (Y≈2) ────────────
# Robot close to actor_0's path, various headings
WP_BATCH1="-2,2,0 -1,2,0 0,1,45 1,1,30 2,1,15 3,1.5,0 4,1.5,345"

# ── Batch 2: Medium range, looking north at multiple actors ──
WP_BATCH2="-3,1.5,90 -1,1.5,90 1,1.5,90 3,1.5,90 5,1.5,90"

# ── Batch 3: Close to west corridor (actor_2 at X≈-4.5) ─────
WP_BATCH3="-3,2,180 -3,3,180 -3,4,180 -3,5,180 -3,3,225 -3,3,135"

# ── Batch 4: East side, looking west (actor_4 at X≈5.5) ─────
WP_BATCH4="4,3,0 4,4,0 4,5,0 4,3,45 4,3,315"

# ── Batch 5: Mid-room panoramic (actors crossing) ───────────
WP_BATCH5="0,3.5,0 0,3.5,45 0,3.5,90 0,3.5,135 0,3.5,180 0,3.5,225 0,3.5,270 0,3.5,315"

# ── Batch 6: North aisle, close to actor_1 (Y≈5.5) ──────────
WP_BATCH6="-2,5,0 0,5,0 2,5,0 -2,5,180 0,5,180 0,4.5,90 0,4.5,270"

# ── Batch 7: Diagonal views (backgrounds with shelves) ──────
WP_BATCH7="-5,2,30 -5,2,60 5,2,120 5,2,150 -5,5,330 5,5,210"

# ── Batch 8: Far range, actors as small targets ──────────────
WP_BATCH8="-5,1.5,0 -5,1.5,45 5,1.5,180 5,1.5,135 0,1,90 -5,6,315 5,6,225"

# ── Batch 9: Center with multiple actors visible ────────────
WP_BATCH9="0,4,0 0,4,90 0,4,180 0,4,270 -1,4,0 1,4,0 -1,4,180 1,4,180"

# ── Batch 10: Additional close-range varied headings ────────
WP_BATCH10="1,2,0 2,2,0 3,2,180 -1,3,270 2,5,270 -4,3,0 4,5,180"

# ── Phase 2 batches: warehouse_actors_varied.sdf ────────────
# Different actor animations (run, talk_a, talk_b, walk mix)
# Actor layout differs from base world — different trajectories

# ── Batch 11 (varied): Close encounters + panoramic ─────────
WP_BATCH11="-1,3,0 0,3,90 1,3,180 2,3,270 -2,4,45 3,4,135 0,5,0 0,5,180"

# ── Batch 12 (varied): West/East corridors with varied actors
WP_BATCH12="-4,3,0 -4,4,45 -4,5,90 4,3,180 4,4,135 4,5,90 0,2,0 0,2,180"

ALL_BATCHES_BASE=(
    "$WP_BATCH1"
    "$WP_BATCH2"
    "$WP_BATCH3"
    "$WP_BATCH4"
    "$WP_BATCH5"
    "$WP_BATCH6"
    "$WP_BATCH7"
    "$WP_BATCH8"
    "$WP_BATCH9"
    "$WP_BATCH10"
)

ALL_BATCHES_VARIED=(
    "$WP_BATCH11"
    "$WP_BATCH12"
)

# Count total waypoints
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
    echo "  Batch $batch_num/$total_batches:  $n_wps waypoints, ~$batch_imgs images [$world_name]"
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
for batch in "${ALL_BATCHES_BASE[@]}"; do
    run_batch "$SDF_BASE" "warehouse_actors" "$batch"
done

# Phase 2: Varied world (mixed animations: run, talk_a, talk_b, walk)
echo ""
echo "========= Phase 2: Varied world ============="
echo "NOTE: The simulation must be restarted with the varied world."
echo "  bash scripts/run_pipeline.sh sim-varied"
echo ""
read -r -p "Press ENTER when the varied simulation is ready (or Ctrl-C to stop)..."

for batch in "${ALL_BATCHES_VARIED[@]}"; do
    run_batch "$SDF_VARIED" "warehouse_actors_varied" "$batch"
done

echo ""
echo "============================================"
echo "  Capture complete!"
echo "============================================"
n_imgs=$(ls "$DATA_DIR/images/" 2>/dev/null | wc -l)
n_xmls=$(ls "$DATA_DIR/labels/" 2>/dev/null | wc -l)
echo "  Images:      $n_imgs"
echo "  Labels (XML): $n_xmls"
echo "  Output:       $DATA_DIR"
echo ""
echo "Next steps:"
echo "  1. Inspect GT:  python3 scripts/verify_gt.py --data_dir $DATA_DIR"
echo "  2. Evaluate:    run the notebook with the new dataset"

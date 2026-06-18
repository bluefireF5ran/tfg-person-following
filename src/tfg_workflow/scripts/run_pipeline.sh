#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<EOF
Uso:
  bash scripts/run_pipeline.sh preflight
  bash scripts/run_pipeline.sh sim
  bash scripts/run_pipeline.sh topics
  bash scripts/run_pipeline.sh check-scan
  bash scripts/run_pipeline.sh move-people [--speed 0.6] [--hz 10]
  bash scripts/run_pipeline.sh capture --topic TOPIC --output_dir DIR [--rate 2.0] [--max_images 300]
  bash scripts/run_pipeline.sh capture-dataset
  bash scripts/run_pipeline.sh verify-gt
  bash scripts/run_pipeline.sh detect   [--model yolo11n.pt] [--conf 0.5]
  bash scripts/run_pipeline.sh follow   [--safe-dist 0.5] [--max-linear 0.3]
  bash scripts/run_pipeline.sh follow-launch [model:=yolo11n.pt] [safe_dist:=0.5]

Ejemplo:
  bash scripts/run_pipeline.sh move-people --speed 0.8
  bash scripts/run_pipeline.sh detect --model yolo11s.pt
  bash scripts/run_pipeline.sh follow --safe-dist 0.6 --max-linear 0.25
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

cmd="$1"
shift

case "$cmd" in
  preflight)
    bash "$ROOT_DIR/scripts/preflight_pipeline.sh"
    ;;

  sim)
    bash "$ROOT_DIR/scripts/run_tb4_sim.sh" "$@"
    ;;

  sim-varied)
    bash "$ROOT_DIR/scripts/run_tb4_sim.sh" warehouse_actors_varied
    ;;

  topics)
    bash "$ROOT_DIR/scripts/discover_sensor_topics.sh"
    ;;

  check-scan)
    set +u
    source /opt/ros/jazzy/setup.bash
    if [[ -f "/home/francisco/tfg_ws/install/setup.bash" ]]; then
      source /home/francisco/tfg_ws/install/setup.bash
    fi
    set -u
    ros2 topic echo /scan --once
    ;;

  move-people)
    python3 "$ROOT_DIR/scripts/move_people.py" "$@"
    ;;

  capture)
    PYTHON_BIN="/home/francisco/tfg_ws/.venv/bin/python"
    if [[ ! -x "$PYTHON_BIN" ]]; then
      echo "[ERROR] No existe Python venv en /home/francisco/tfg_ws/.venv"
      exit 1
    fi

    set +u
    source /opt/ros/jazzy/setup.bash
    if [[ -f "/home/francisco/tfg_ws/install/setup.bash" ]]; then
      source /home/francisco/tfg_ws/install/setup.bash
    fi
    set -u

    "$PYTHON_BIN" "$ROOT_DIR/scripts/capture_images_ros2.py" "$@"
    ;;

  detect)
    set +u
    source /opt/ros/jazzy/setup.bash
    set -u
    python3 "$ROOT_DIR/scripts/person_detector.py" "$@"
    ;;

  follow)
    set +u
    source /opt/ros/jazzy/setup.bash
    set -u
    python3 "$ROOT_DIR/scripts/person_follower.py" "$@"
    ;;

  follow-launch)
    set +u
    source /opt/ros/jazzy/setup.bash
    set -u
    ros2 launch "$ROOT_DIR/launch/person_follow.launch.py" "$@"
    ;;

  capture-dataset)
    bash "$ROOT_DIR/scripts/capture_dataset.sh" "$@"
    ;;

  verify-gt)
    python3 "$ROOT_DIR/scripts/verify_gt.py" --data_dir "$ROOT_DIR/data/sim_dataset" "$@"
    ;;

  *)
    usage
    exit 1
    ;;
esac

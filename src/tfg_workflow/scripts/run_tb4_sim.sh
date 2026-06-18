#!/usr/bin/env bash
set -euo pipefail

set +u
source /opt/ros/jazzy/setup.bash

if [[ -f "/home/francisco/tfg_ws/install/setup.bash" ]]; then
  source /home/francisco/tfg_ws/install/setup.bash
fi
set -u

export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

WORLD="${1:-warehouse_actors}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_DIR="$(cd "$SCRIPT_DIR/../launch" && pwd)"
WORLDS_DIR="$(cd "$SCRIPT_DIR/../worlds" && pwd)"

echo "[INFO] Mundo: $WORLD"
echo "[INFO] Lanzando simulación TurtleBot4 en Gazebo..."

if [[ -f "$WORLDS_DIR/${WORLD}.sdf" ]]; then
  echo "[INFO] Usando mundo personalizado: $WORLDS_DIR/${WORLD}.sdf"
  # yaw=0 (mira al ESTE): el objetivo se acerca de frente por el este, asi el robot lo ve
  # CENTRADO y de forma continua -> deteccion fiable y fijado rapido. (Con yaw=pi el robot
  # nacia de espaldas al objetivo y solo lo cazaba de refilon al girar -> casi no lo detectaba.)
  # El dock queda 0.157 m delante (este), pero con safety_override=full los reflejos del
  # Create3 estan desactivados, asi que el robot sale de el como en un undock normal.
  ros2 launch "$LAUNCH_DIR/sim_with_people.launch.py" world:="$WORLD" x:=-2.0 y:=2.0 yaw:=0.0
else
  echo "[INFO] Usando mundo del sistema: $WORLD"
  exec ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py world:="$WORLD"
fi

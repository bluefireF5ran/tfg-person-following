#!/usr/bin/env bash
: "${TFG_WS:=$HOME/tfg_ws}"   # raiz del workspace; exportala si usas otra
set -euo pipefail

set +u
source /opt/ros/jazzy/setup.bash

if [[ -f "${TFG_WS}/install/setup.bash" ]]; then
  source ${TFG_WS}/install/setup.bash
fi
set -u

export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# GPU NVIDIA (PRIME render offload): fuerza a que Gazebo se ejecute en la GPU. Sin esto, en
# portatiles con grafica hibrida el GpuLidar falla y /scan devuelve un valor FIJO (el laser no
# "ve" obstaculos ni personas). Inocuo en equipos solo-NVIDIA. Verificar con `nvidia-smi`
# (deben aparecer `gz sim server`/`gui`) y que /scan de rangos VARIABLES.
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
# En esta maquina las dos de arriba NO bastaban: glvnd elegia Mesa (warning `libEGL ... driver
# (null)`) y el GpuLidar no renderizaba. Hay que forzar tambien el vendor EGL de NVIDIA. El
# fichero existe si esta el driver glvnd de NVIDIA (junto a 50_mesa.json); inocuo si ya se usa.
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"

# Por defecto, el circuito cerrado con actores que se usa para la demo de seguimiento. Los mundos
# warehouse_actors*.sdf antiguos (captura de dataset) están archivados en worlds/_archivo/.
WORLD="${1:-warehouse_unified}"
# Pose de spawn del robot: por defecto (-2,2,0) ---la usada para grabar el vídeo---, pero
# configurable por entorno para que la campaña pueda VARIAR el punto de partida en cada ejecución
# (cada run parte distinta -> repeticiones con sentido estadístico, no el mismo escenario repetido).
SPAWN_X="${SPAWN_X:--2.0}"
SPAWN_Y="${SPAWN_Y:-2.0}"
SPAWN_YAW="${SPAWN_YAW:-0.0}"
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
  echo "[INFO] Spawn robot: x=$SPAWN_X y=$SPAWN_Y yaw=$SPAWN_YAW"
  ros2 launch "$LAUNCH_DIR/sim_with_people.launch.py" world:="$WORLD" x:="$SPAWN_X" y:="$SPAWN_Y" yaw:="$SPAWN_YAW"
else
  echo "[INFO] Usando mundo del sistema: $WORLD"
  exec ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py world:="$WORLD"
fi

#!/usr/bin/env bash
: "${TFG_WS:=$HOME/tfg_ws}"   # raiz del workspace; exportala si usas otra
set -euo pipefail

echo "[INFO] Configuración recomendada para TurtleBot4 + ROS 2 Jazzy + Gazebo Harmonic"

ROS_SETUP="/opt/ros/jazzy/setup.bash"
WS_SETUP="${TFG_WS}/install/setup.bash"

if [[ ! -f "$ROS_SETUP" ]]; then
  echo "[ERROR] No existe $ROS_SETUP"
  exit 1
fi

echo "[INFO] Aplicando variables para esta terminal..."
set +u
source "$ROS_SETUP"

if [[ -f "$WS_SETUP" ]]; then
  source "$WS_SETUP"
fi
set -u

export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

echo ""
echo "[OK] Variables activas en esta sesión:"
echo "  ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "  RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "  __NV_PRIME_RENDER_OFFLOAD=$__NV_PRIME_RENDER_OFFLOAD"
echo "  __GLX_VENDOR_LIBRARY_NAME=$__GLX_VENDOR_LIBRARY_NAME"

echo ""
echo "[INFO] Para persistir en ~/.bashrc añade:"
cat <<'EOF'
source /opt/ros/jazzy/setup.bash
source ${TFG_WS}/install/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
EOF

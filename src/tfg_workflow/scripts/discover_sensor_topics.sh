#!/usr/bin/env bash
: "${TFG_WS:=$HOME/tfg_ws}"   # raiz del workspace; exportala si usas otra
set -euo pipefail

set +u
source /opt/ros/jazzy/setup.bash
if [[ -f "${TFG_WS}/install/setup.bash" ]]; then
  source ${TFG_WS}/install/setup.bash
fi
set -u

echo "[INFO] Topics de imagen disponibles:"
ros2 topic list | grep image || echo "  (ninguno por ahora)"

echo ""
echo "[INFO] Topics de láser disponibles:"
ros2 topic list | grep scan || echo "  (ninguno por ahora)"

echo ""
echo "[INFO] Tipo de topic /scan (si existe):"
ros2 topic type /scan || true

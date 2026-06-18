#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="python3"

echo "[1/4] Ejecutables"
command -v ros2 >/dev/null && echo "  OK ros2" || { echo "  ERROR ros2 no encontrado"; exit 1; }
command -v nvidia-smi >/dev/null && echo "  OK nvidia-smi" || echo "  WARN nvidia-smi no encontrado"

echo "[2/4] ROS packages TurtleBot4"
set +u
source /opt/ros/jazzy/setup.bash
if [[ -f "/home/francisco/tfg_ws/install/setup.bash" ]]; then
  source /home/francisco/tfg_ws/install/setup.bash
fi
set -u

required_pkgs=(turtlebot4_gz_bringup turtlebot4_viz turtlebot4_simulator)
for pkg in "${required_pkgs[@]}"; do
  if ros2 pkg list | grep -qx "$pkg"; then
    echo "  OK $pkg"
  else
    echo "  ERROR falta paquete: $pkg"
    exit 1
  fi
done

echo "[3/4] Python deps"
"$PYTHON_BIN" -c "import importlib.util as u;mods=['rclpy','cv2','ultralytics','fiftyone','torch'];missing=[m for m in mods if not u.find_spec(m)];print('  OK deps' if not missing else '  ERROR faltan: '+', '.join(missing));exit(0 if not missing else 1)"

echo "[4/4] GPU process snapshot"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi | egrep "gz sim server|gz sim gui|Process Name|MiB" || true
fi

echo "[OK] Preflight completado"

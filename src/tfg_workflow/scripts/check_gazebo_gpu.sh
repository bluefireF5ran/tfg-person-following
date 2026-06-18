#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi no está disponible en este sistema"
  exit 1
fi

echo "[INFO] Procesos de Gazebo usando GPU NVIDIA:"
nvidia-smi | egrep "gz sim server|gz sim gui|Process Name|MiB" || true

echo ""
echo "[INFO] Si no aparecen 'gz sim server/gui', Gazebo podría estar en CPU."

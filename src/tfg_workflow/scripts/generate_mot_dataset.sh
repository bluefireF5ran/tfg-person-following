#!/usr/bin/env bash
# ===========================================================================
# generate_mot_dataset.sh — Orquesta la generación del dataset de TRACKING
# multi-objeto (formato MOTChallenge) desde warehouse_unified.
#
# Es el último encargo del tutor: etiquetar un dataset de personas simuladas
# con un ID estable por persona, para evaluar tracking.
#
# REQUISITO: la simulación debe estar YA corriendo en otra terminal:
#     bash src/tfg_workflow/scripts/run_tb4_sim.sh warehouse_unified
#
# Uso:
#     bash src/tfg_workflow/scripts/generate_mot_dataset.sh [OUT_DIR] [FRAMES_PER_SEQ]
#
# Ejemplos:
#     bash src/tfg_workflow/scripts/generate_mot_dataset.sh
#     bash src/tfg_workflow/scripts/generate_mot_dataset.sh data/mot_dataset 200
#
# Variables de entorno opcionales:
#     VIEWPOINTS   poses del robot "x,y,yaw  x,y,yaw ..." (se teletransporta a cada una)
#     RATE         Hz de captura (def. 10)
#     HALF_WIDTH   semiancho del box 3D del actor (def. 0.18, cajas ceñidas)
#     NO_DRAW=1    desactiva el guardado de imágenes de verificación
# ===========================================================================
set -euo pipefail

# ── Entorno ROS 2 ──────────────────────────────────────────────────────────
set +u
source /opt/ros/jazzy/setup.bash
if [[ -f "/home/francisco/tfg_ws/install/setup.bash" ]]; then
  source /home/francisco/tfg_ws/install/setup.bash
fi
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PY="$SCRIPT_DIR/capture_mot_dataset.py"

# ── Parámetros ─────────────────────────────────────────────────────────────
OUT_DIR="${1:-$WS_DIR/data/mot_dataset}"
FRAMES_PER_SEQ="${2:-200}"
# Mundo objetivo (env WORLD). El nombre del <world> interno SIEMPRE es
# 'warehouse_unified' (las variantes reusan ese nombre), por eso WORLD_NAME fijo.
WORLD_FILE="${WORLD:-warehouse_unified}"
SDF="$(cd "$SCRIPT_DIR/../worlds" && pwd)/${WORLD_FILE}.sdf"
WORLD="warehouse_unified"
RATE="${RATE:-10}"
HALF_WIDTH="${HALF_WIDTH:-0.18}"
SETTLE="${SETTLE:-2.0}"
OCC_MARGIN="${OCC_MARGIN:-0.5}"
# Viewpoints por defecto: 5 poses que cubren la zona de paso de los actores.
# El espacio inicial es OBLIGATORIO: evita que argparse tome '-2,...' como flag.
VIEWPOINTS="${VIEWPOINTS:- -2,0,90  2,0,110  0,-2,80  -3,1,60  3,1,120 }"

DRAW_FLAG="--draw"
[[ "${NO_DRAW:-0}" == "1" ]] && DRAW_FLAG=""

# Guardar profundidad + meta.jsonl por secuencia (ON por defecto): permite re-afinar
# el refinado de cajas OFFLINE con tune_refine.py sin volver a lanzar Gazebo.
SAVE_DEPTH_FLAG="--save_depth"
[[ "${SAVE_DEPTH:-1}" == "0" ]] && SAVE_DEPTH_FLAG=""

# Filtro de oclusión por profundidad (recomendado, ON por defecto): descarta
# actores tapados por estanterías u otros actores usando la imagen de profundidad.
DEPTH_FLAG="--use_depth --occ_margin $OCC_MARGIN"
[[ "${USE_DEPTH:-1}" == "0" ]] && DEPTH_FLAG=""

# Refinado de cajas por profundidad (ON por defecto): ciñe cada caja a la silueta
# real del actor -> cajas centradas y sin cajas 'vacías' (feedback de Eugenio).
[[ "${REFINE:-1}" == "1" ]] && DEPTH_FLAG="$DEPTH_FLAG --depth_refine"

# Box por malla (Plan B): si existe el JSON de calibración, se usa.
# OJO: va FUERA del dataset (data/, no data/mot_dataset/) para que el borrado
# de la salida no lo elimine antes de la captura.
BOX_CONFIG="${BOX_CONFIG:-$WS_DIR/data/actor_box_config.json}"
BOX_FLAG=""
[[ -f "$BOX_CONFIG" ]] && BOX_FLAG="--box_config $BOX_CONFIG"

echo "============================================================"
echo "  Generador de dataset de tracking multi-ID (MOTChallenge)"
echo "============================================================"
echo "  Mundo        : $WORLD"
echo "  SDF          : $SDF"
echo "  Salida       : $OUT_DIR"
echo "  Viewpoints   :$VIEWPOINTS"
echo "  Frames/seq   : $FRAMES_PER_SEQ   Rate: ${RATE} Hz"
echo "  Box actor    : half_width=$HALF_WIDTH"
echo "  Filtro oclus.: ${DEPTH_FLAG:-(desactivado)}"
echo "  Box por malla: ${BOX_FLAG:-(global, sin calibrar)}"
echo "============================================================"

# ── Comprobaciones previas ─────────────────────────────────────────────────
fail() { echo -e "\n[ERROR] $1" >&2; exit 1; }

[[ -f "$SDF" ]] || fail "No existe el SDF: $SDF"
[[ -f "$PY"  ]] || fail "No existe el script de captura: $PY"

echo "[CHECK] Instancias de Gazebo activas..."
N_GZ="$(pgrep -fc 'gz sim server' 2>/dev/null || echo 0)"
if [[ "$N_GZ" -eq 0 ]]; then
  fail "No hay Gazebo corriendo. Lanza primero:\n    bash $SCRIPT_DIR/run_tb4_sim.sh warehouse_unified"
elif [[ "$N_GZ" -gt 1 ]]; then
  echo "[WARN] Hay $N_GZ instancias de 'gz sim server'. Riesgo de poses inconsistentes."
  echo "       Recomendado: pkill -9 -f 'gz sim server'; pkill -f parameter_bridge; y relanzar UNA sola."
  read -r -p "       ¿Continuar de todos modos? [y/N] " ans
  [[ "${ans,,}" == "y" ]] || fail "Abortado por el usuario."
else
  echo "[CHECK]   OK: 1 instancia de Gazebo."
fi

echo "[CHECK] Topic de cámara publicando (/oakd/rgb/preview/image_raw)..."
# El bridge de cámara usa QoS de SENSOR (best-effort); 'ros2 topic echo' (reliable)
# da falsos negativos. Comprobamos con un subscriber rclpy de QoS de sensor.
if ! timeout 35 python3 - <<'PYCHK'
import sys, rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
rclpy.init()
n = rclpy.create_node("cam_check")
got = {"ok": False}
n.create_subscription(Image, "/oakd/rgb/preview/image_raw",
                      lambda m: got.update(ok=True), qos_profile_sensor_data)
import time
end = time.time() + 30
while time.time() < end and not got["ok"]:
    rclpy.spin_once(n, timeout_sec=0.3)
rclpy.shutdown()
sys.exit(0 if got["ok"] else 1)
PYCHK
then
  fail "No llegan imágenes en /oakd/rgb/preview/image_raw. ¿Está la simulación cargada del todo? Espera ~30 s tras lanzarla."
fi
echo "[CHECK]   OK: cámara activa."

# ── Limpieza de salida previa ──────────────────────────────────────────────
if [[ -d "$OUT_DIR" ]] && [[ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]]; then
  echo "[WARN] El directorio de salida ya existe y no está vacío: $OUT_DIR"
  read -r -p "       ¿Borrar su contenido y regenerar? [y/N] " ans
  if [[ "${ans,,}" == "y" ]]; then
    rm -rf "${OUT_DIR:?}/"*
    echo "[INFO] Contenido anterior borrado."
  else
    fail "Abortado para no mezclar secuencias antiguas y nuevas."
  fi
fi
mkdir -p "$OUT_DIR"

# ── Captura ────────────────────────────────────────────────────────────────
echo -e "\n[RUN] Generando dataset (esto teletransporta el robot a cada viewpoint)..."
python3 "$PY" \
  --sdf "$SDF" --world "$WORLD" \
  --output_dir "$OUT_DIR" \
  --viewpoints "$VIEWPOINTS" \
  --frames_per_seq "$FRAMES_PER_SEQ" \
  --rate "$RATE" --settle_time "$SETTLE" \
  --actor_half_width "$HALF_WIDTH" \
  $DRAW_FLAG $DEPTH_FLAG $BOX_FLAG $SAVE_DEPTH_FLAG

# ── Resumen ────────────────────────────────────────────────────────────────
echo -e "\n============================================================"
echo "  RESUMEN DEL DATASET GENERADO"
echo "============================================================"
TOTAL_IMG=0; TOTAL_GT=0
for s in "$OUT_DIR"/seq*; do
  [[ -d "$s" ]] || continue
  gt="$s/gt/gt.txt"
  n_img=$(ls "$s/img1" 2>/dev/null | wc -l)
  n_gt=$(wc -l < "$gt" 2>/dev/null || echo 0)
  ids=$(cut -d, -f2 "$gt" 2>/dev/null | sort -un | tr '\n' ' ')
  printf "  %-8s  %3d frames  %4d filas GT  IDs: %s\n" "$(basename "$s")" "$n_img" "$n_gt" "$ids"
  TOTAL_IMG=$((TOTAL_IMG + n_img)); TOTAL_GT=$((TOTAL_GT + n_gt))
done
echo "------------------------------------------------------------"
echo "  TOTAL: $TOTAL_IMG imágenes, $TOTAL_GT filas de GT"
echo "  id_map.json: $OUT_DIR/id_map.json"
[[ -n "$DRAW_FLAG" ]] && echo "  Verificación visual: $OUT_DIR/seqNN/verify/*.jpg"
echo "============================================================"
echo "[OK] Dataset listo. Revisa las imágenes de verify/ antes de usarlo."
echo "     Siguiente paso (evaluar trackers):"
echo "       python3 $SCRIPT_DIR/track_persons.py  --source $OUT_DIR/seq01/img1 --model yolo11s.pt --tracker bytetrack --out /tmp/bt.txt"
echo "       python3 $SCRIPT_DIR/evaluate_mot.py   --gt $OUT_DIR/seq01/gt/gt.txt --pred ByteTrack=/tmp/bt.txt --filter-valid"

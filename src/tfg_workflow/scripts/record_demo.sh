#!/bin/bash
# record_demo.sh — Graba DOS vistas del MISMO circuito a la vez y las recorta en 2 videos sincronizados.
#
# Idea: capturar de una sola pasada la region que abarca 2 monitores contiguos (misma linea temporal
# => exactamente el mismo circuito), y luego recortar esa grabacion en dos MP4 independientes.
#
# DISPOSICION DE MONITORES (de `xrandr`, ajustar OFFSETS abajo si cambia):
#   DP-0   1920x1080 +0+0     (IZQUIERDA)  -> aqui GAZEBO (vista externa)
#   DP-4   1920x1080 +1920+0  (CENTRO)     -> aqui el POV  (/person_detections/image, objetivo en verde)
#   HDMI-0 1920x1080 +3840+0  (DERECHA)
#
# USO:
#   1) Lanza la simulacion y el seguimiento (run_tb4_sim.sh + person_follow.launch.py).
#   2) Maximiza GAZEBO en el monitor IZQUIERDO y el POV (rqt_image_view /person_detections/image)
#      en el CENTRAL.
#   3) bash scripts/record_demo.sh           # empieza a grabar
#   4) Cuando termine el circuito, pulsa 'q' (o Ctrl-C) en esta terminal para parar.
#   5) El script recorta automaticamente en gazebo_<ts>.mp4 y pov_<ts>.mp4.
#
# Si pones las ventanas en OTROS monitores, cambia SPAN_X/SPAN_W y los CROP_* de abajo.

set -e

OUT_DIR="${1:-$HOME/tfg_ws/demo_videos}"
FPS="${FPS:-30}"
mkdir -p "$OUT_DIR"
TS=$(date +%Y%m%d_%H%M%S)
RAW="$OUT_DIR/raw_${TS}.mp4"

# Display X a capturar (p.ej. :1 en este equipo). Se usa $DISPLAY; si no, :0 por defecto.
DISP="${DISPLAY:-:0}"

# Region a capturar: 2 monitores contiguos (DP-0 + DP-4) = 3840x1080 desde el origen +0+0.
SPAN_W=3840
SPAN_H=1080
SPAN_X=0
SPAN_Y=0

# Recortes (dentro de la grabacion RAW): mitad izquierda = Gazebo, mitad derecha = POV.
CROP_GAZEBO="crop=1920:1080:0:0"
CROP_POV="crop=1920:1080:1920:0"

echo "============================================================"
echo " Grabando ${DISP} region ${SPAN_W}x${SPAN_H} en +${SPAN_X},${SPAN_Y} a ${FPS} fps"
echo "   IZQUIERDA -> Gazebo   |   CENTRO -> POV (cajas)"
echo " Pulsa 'q' (o Ctrl-C) para PARAR cuando acabe el circuito."
echo "============================================================"

# Captura en vivo con preset ultrafast para no robar CPU a Gazebo/YOLO; se reencoda al recortar.
ffmpeg -hide_banner -loglevel warning \
  -f x11grab -framerate "$FPS" -video_size "${SPAN_W}x${SPAN_H}" -i "${DISP}+${SPAN_X},${SPAN_Y}" \
  -c:v libx264 -preset ultrafast -crf 18 -pix_fmt yuv420p "$RAW" || true

if [ ! -s "$RAW" ]; then
  echo ""
  echo "ERROR: no se grabó nada ($RAW vacío o inexistente). Revisa DISPLAY/region y reintenta."
  exit 1
fi

echo ""
echo "Recortando en 2 videos sincronizados..."
ffmpeg -hide_banner -loglevel warning -y -i "$RAW" -filter:v "$CROP_GAZEBO" \
  -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p "$OUT_DIR/gazebo_${TS}.mp4"
ffmpeg -hide_banner -loglevel warning -y -i "$RAW" -filter:v "$CROP_POV" \
  -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p "$OUT_DIR/pov_${TS}.mp4"

echo ""
echo "LISTO:"
echo "  Gazebo: $OUT_DIR/gazebo_${TS}.mp4"
echo "  POV:    $OUT_DIR/pov_${TS}.mp4"
echo "  (RAW combinado: $RAW  — borralo si no lo necesitas)"

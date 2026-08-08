#!/usr/bin/env bash
# run_eval_campaign.sh — Campaña de evaluación NO interactiva: repite N ejecuciones de
# DUR segundos lanzando, en cada una, la simulación (run_tb4_sim.sh) y el seguimiento
# (person_follow.launch.py), y cerrando todo limpio entre iteraciones.
#
# Acumula varias ejecuciones del MISMO circuito para reforzar la campaña experimental (cap. 7).
# Cada ejecución registra su telemetría en W&B (wandb/offline-run-*), que se agrega luego con
# analyze_telemetry.py.
#
# USO:
#   bash src/tfg_workflow/scripts/run_eval_campaign.sh [N] [DUR_S]
#     N      = nº de ejecuciones        (def. 3)
#     DUR_S  = duración de cada una (s)  (def. 120 = 2 min)
#   Env opcionales: STARTUP_MAX (espera máx. de arranque de Gazebo, def. 120 s),
#                   SETTLE_S (cierre entre iteraciones, def. 10), WORLD (def. warehouse_unified).
#
# Necesita display (Gazebo con GUI) y GPU. Deja esta terminal a la vista: verás el arranque de
# Gazebo y, cuando publique la cámara, el lanzamiento del seguimiento. Logs en
# logs_campaign/run_AAAAMMDD_HHMMSS/ (una subcarpeta nueva por cada invocación).
: "${TFG_WS:=$HOME/tfg_ws}"   # raiz del workspace; exportala si usas otra

set -u
N="${1:-3}"
DUR="${2:-120}"
STARTUP_MAX="${STARTUP_MAX:-120}"
# Retardo tras detectar la cámara y ANTES de lanzar el follower. Reproduce el flujo manual (donde
# tardas unos segundos en lanzar la 2ª terminal): si el follower arranca demasiado pronto, el
# instante del cruce del actor coincide con el robot y puede robarse el objetivo. Subir si pasa.
FOLLOWER_DELAY_S="${FOLLOWER_DELAY_S:-15}"
SETTLE_S="${SETTLE_S:-10}"
# Variación del punto de partida del robot en CADA ejecución (1=sí, 0=no). Así cada run parte de
# una pose distinta -> las repeticiones no son el mismo escenario determinista, sino muestras con
# condiciones iniciales diferentes (geometría de adquisición y momento de los cruces distintos).
RANDOMIZE_SPAWN="${RANDOMIZE_SPAWN:-1}"
WORLD="${WORLD:-warehouse_unified}"
CAM_TOPIC="/oakd/rgb/preview/image_raw"   # tópico que confirma que Gazebo + sensores están vivos

ROOT="${TFG_WS}"
SCRIPTS="$ROOT/src/tfg_workflow/scripts"
LAUNCH="$ROOT/src/tfg_workflow/launch"
# Subcarpeta propia por invocación (con fecha/hora): así los logs de campañas anteriores NO se
# sobrescriben nunca. Cada lanzamiento del script crea su run_AAAAMMDD_HHMMSS/.
LOGDIR="$ROOT/logs_campaign/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

# IMPORTANTE: los setup de ROS referencian variables no definidas; con `set -u` activo, el
# `source` aborta el script (creaba logs_campaign vacío y no generaba logs). Se desactiva `set -u`
# alrededor del source (igual que hace run_tb4_sim.sh) y se reactiva después.
set +u
source /opt/ros/jazzy/setup.bash 2>/dev/null || true
[ -f "$ROOT/install/setup.bash" ] && source "$ROOT/install/setup.bash" 2>/dev/null || true
set -u

SIM_PGID=""; FOLL_PGID=""

kill_group() { [ -n "$1" ] && kill -INT -- "-$1" 2>/dev/null; }

hard_cleanup() {
  for pat in "person_follow.launch.py" "person_follower.py" "person_detector.py" \
             "sim_with_people.launch.py" "parameter_bridge" "gz sim" "ruby .*gz"; do
    pkill -INT -f "$pat" 2>/dev/null
  done
}

cleanup_iter() {
  echo "[campaign] cerrando iteración..."
  kill_group "$FOLL_PGID"; sleep 3
  kill_group "$SIM_PGID";  sleep 3
  hard_cleanup; sleep "$SETTLE_S"
  FOLL_PGID=""; SIM_PGID=""
}

trap 'echo; echo "[campaign] interrumpido, limpiando..."; cleanup_iter; exit 130' INT TERM

# Espera activa: vuelve 0 si CAM_TOPIC aparece antes de STARTUP_MAX, 1 si no.
wait_for_sim() {
  local waited=0
  while [ "$waited" -lt "$STARTUP_MAX" ]; do
    if ros2 topic list 2>/dev/null | grep -q "$CAM_TOPIC"; then return 0; fi
    sleep 3; waited=$((waited+3))
    printf '\r[campaign] esperando a Gazebo (%ss/%ss)...' "$waited" "$STARTUP_MAX"
  done
  echo; return 1
}

echo "============================================================"
echo " Campaña: $N ejecuciones de $DUR s | mundo=$WORLD | arranque máx=$STARTUP_MAX s"
echo "============================================================"

for i in $(seq 1 "$N"); do
  echo "===== Iteración $i/$N ($(date +%H:%M:%S)) ====="

  # --- Punto de partida del robot: aleatorio por iteración (si RANDOMIZE_SPAWN=1) ---
  # Rango cómodo alrededor de (-2,2,0): distancia inicial al objetivo y encuadre distintos en cada
  # run, pero dentro del rango de adquisición para que siga fijando al objetivo.
  if [ "$RANDOMIZE_SPAWN" = "1" ]; then
    read SPAWN_X SPAWN_Y SPAWN_YAW < <(LC_ALL=C awk -v s="$RANDOM$i" 'BEGIN{srand(s);
      printf "%.2f %.2f %.2f", -2.4+rand()*0.7, 1.6+rand()*0.8, -0.25+rand()*0.5}')
    export SPAWN_X SPAWN_Y SPAWN_YAW
    echo "[campaign] spawn aleatorio: x=$SPAWN_X y=$SPAWN_Y yaw=$SPAWN_YAW"
  else
    unset SPAWN_X SPAWN_Y SPAWN_YAW
  fi

  # --- Simulación (grupo propio con setsid; hereda SPAWN_* exportadas arriba) ---
  echo "[campaign] lanzando simulación (log: $LOGDIR/sim_$i.log)..."
  setsid bash "$SCRIPTS/run_tb4_sim.sh" "$WORLD" >"$LOGDIR/sim_$i.log" 2>&1 &
  SIM_PGID=$!

  if ! wait_for_sim; then
    echo "[campaign] ERROR: Gazebo no publicó $CAM_TOPIC en ${STARTUP_MAX}s. Últimas líneas del log:"
    echo "------------------------------------------------------------"
    tail -n 25 "$LOGDIR/sim_$i.log"
    echo "------------------------------------------------------------"
    cleanup_iter
    echo "[campaign] abortando. Revisa $LOGDIR/sim_$i.log (¿GPU/EGL?, ¿display?, ¿mundo?)."
    exit 1
  fi
  echo; echo "[campaign] Gazebo listo (cámara publicando). Esperando ${FOLLOWER_DELAY_S}s antes del seguimiento (imita el flujo manual)..."
  sleep "$FOLLOWER_DELAY_S"

  # --- Seguimiento (detector + follower) ---
  echo "[campaign] lanzando seguimiento (log: $LOGDIR/foll_$i.log)..."
  setsid ros2 launch "$LAUNCH/person_follow.launch.py" >"$LOGDIR/foll_$i.log" 2>&1 &
  FOLL_PGID=$!
  echo "[campaign] corriendo ${DUR}s (Ctrl-C para abortar la campaña)..."
  sleep "$DUR"

  cleanup_iter
  echo "[campaign] iteración $i completa."
done

trap - INT TERM
echo "============================================================"
echo " Campaña completa: $N ejecuciones de $DUR s."
echo " Logs:        $LOGDIR/sim_*.log , $LOGDIR/foll_*.log"
echo " Telemetría:  $ROOT/wandb/offline-run-*  (agrega con analyze_telemetry.py)"
echo "============================================================"

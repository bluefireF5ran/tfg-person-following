#!/bin/bash
# Script para ejecutar simulación con variedad de actores
# Mejora el realismo visual con diferentes alturas y comportamientos
: "${TFG_WS:=$HOME/tfg_ws}"   # raiz del workspace; exportala si usas otra

set -e

WORKSPACE_ROOT="${TFG_WS}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  👥 SIMULACIÓN CON VARIEDAD DE ACTORES                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 Características de variedad:"
echo "   ✅ 10 actores con diferentes características"
echo "   ✅ Alturas variables: 0.9m - 1.1m (bajo/alto)"
echo "   ✅ Velocidades variables: 0.8 - 3.0 m/s"
echo "   ✅ Animaciones variadas: walk, run, talk"
echo "   ✅ Comportamientos: individual, parejas, grupos"
echo ""
echo "👥 Distribución de actores:"
echo "   • Actor 0: Primary target (estatura media)"
echo "   • Actor 1: Persona alta corriendo"
echo "   • Actor 2: Persona baja caminando lento"
echo "   • Actor 3-5: Pareja conversando"
echo "   • Actor 6: Persona muy alta corriendo rápido"
echo "   • Actor 7: Persona baja caminata intermitente"
echo "   • Actor 8-9: Nuevos actores con características únicas"
echo "   • Actor 10: Persona baja hablando sola"
echo ""
echo "🚀 Iniciando simulación con variedad..."
echo ""

cd "$WORKSPACE_ROOT"
source install/setup.bash

# Launch simulation with variety
ros2 launch src/tfg_workflow/launch/sim_with_people.launch.py \
    world:=warehouse_actors_variety \
    rviz:=false \
    x:=-2.0 \
    y:=2.0 \
    yaw:=0.0

echo ""
echo "✅ Simulación con variedad iniciada"
echo ""
echo "💡 Tips:"
echo "   - Observa las diferentes alturas de los actores"
echo "   - Nota las velocidades variables (caminar vs correr)"
echo "   - La pareja conversando (Actors 3-5) ahora tiene contexto"
echo "   - Mayor diversidad visual = mayor realismo"
echo ""
echo "🔧 Para ejecutar person-following:"
echo "   cd ${TFG_WS}/src/tfg_workflow"
echo "   bash scripts/run_pipeline.sh follow-launch"
echo ""

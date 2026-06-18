#!/bin/bash
# Script para ejecutar el circuito de testing controlado
# PERFECTO para defensa del TFG

set -e

WORKSPACE_ROOT="/home/francisco/tfg_ws"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🧪 CIRCUITO DE TESTING - DEFENSA TFG                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 Características del circuito:"
echo "   ✅ 1 Robot + 1 Agente objetivo"
echo "   ✅ Obstáculos estratégicos: pilares, cajas, barreras"
echo "   ✅ Recorrido rectangular predecible"
echo "   ✅ Zonas de test marcadas con colores"
echo "   ✅ Diversión: ~20 segundos por vuelta"
echo ""
echo "🎯 Zonas de test del circuito:"
echo "   🟢 START (Verde): Inicio del circuito"
echo "   🟠 TEST 1 (Naranja): Seguimiento en recta"
echo "   🔴 TEST 2 (Rojo): Navegación de curvas"
echo "   🔵 TEST 4 (Morado): Recuperación de oclusiones"
echo ""
echo "🚀 Iniciando circuito de testing..."
echo ""

cd "$WORKSPACE_ROOT"
source install/setup.bash

# Launch simulation with testing circuit (versión ultra simple)
ros2 launch src/tfg_workflow/launch/sim_with_people.launch.py \
    world:=testing_circuit_ultra_simple \
    rviz:=false \
    x:=-2.0 \
    y:=0.0 \
    yaw:=0.0

echo ""
echo "✅ Circuito de testing iniciado"
echo ""
echo "💡 Para ejecutar person-following:"
echo "   cd /home/francisco/tfg_ws/src/tfg_workflow"
echo "   bash scripts/run_pipeline.sh follow-launch"
echo ""
echo "🎯 El agente seguirá un circuito rectangular predecible"
echo "   perfecto para demostrar el sistema en tu defensa."
echo ""

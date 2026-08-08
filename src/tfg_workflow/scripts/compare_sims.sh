#!/bin/bash
# Script para comparar visualmente ambos mundos
# Uso: bash compare_sims.sh
: "${TFG_WS:=$HOME/tfg_ws}"   # raiz del workspace; exportala si usas otra

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🔍 Comparativa Visual: Original vs Realista                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Este script te ayudará a comparar ambos mundos visualmente."
echo ""
echo "📋 Diferencias clave:"
echo "   ❌ Original: Sombras desactivadas, movimiento lento (0.36 m/s)"
echo "   ✅ Realista: Sombras activadas, movimiento realista (1.4-3.0 m/s)"
echo ""
echo "🚀 Opciones de ejecución:"
echo ""
echo "1. Ejecutar MUNDO ORIGINAL (lento, sin sombras):"
echo "   cd ${TFG_WS}"
echo "   source install/setup.bash"
echo "   ros2 launch src/tfg_workflow/launch/sim_with_people.launch.py world:=warehouse_actors"
echo ""
echo "2. Ejecutar MUNDO REALISTA (velocidad normal, con sombras):"
echo "   cd ${TFG_WS}/src/tfg_workflow"
echo "   bash scripts/run_realistic_sim.sh"
echo ""
echo "💡 Recomendación:"
echo "   Abre dos terminales y ejecuta UNO en cada terminal para"
echo "   comparar en tiempo real las diferencias visuales."
echo ""
echo "🎯 Para test person-following con el mundo realista:"
echo "   # Terminal 1: Iniciar simulación realista"
echo "   cd ${TFG_WS}/src/tfg_workflow"
echo "   bash scripts/run_realistic_sim.sh"
echo ""
echo "   # Terminal 2: Ejecutar person-following"
echo "   cd ${TFG_WS}/src/tfg_workflow"
echo "   bash scripts/run_pipeline.sh follow-launch"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "¿Quieres ejecutar el mundo realista ahora? (s/n)"
read -r response

if [[ "$response" =~ ^([sS][iI]|[sS]|y|yes)$ ]]; then
    echo ""
    echo "🚀 Iniciando mundo realista..."
    cd ${TFG_WS}/src/tfg_workflow
    bash scripts/run_realistic_sim.sh
else
    echo ""
    echo "📚 Para más información, consulta: MEJORAS_VISUALES_GAZEBO.md"
fi

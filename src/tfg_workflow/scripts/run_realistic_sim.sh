#!/bin/bash
# Script para ejecutar simulación con mundo visualmente mejorado
# Uso: bash run_realistic_sim.sh
: "${TFG_WS:=$HOME/tfg_ws}"   # raiz del workspace; exportala si usas otra

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🎨 TurtleBot4 Simulación - MUNDO VISUALMENTE MEJORADO       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 Características del mundo mejorado:"
echo "   ✅ Sombras activadas para realismo"
echo "   ✅ Velocidades realistas: 1.4-3.0 m/s (vs 0.36 m/s anterior)"
echo "   ✅ Animaciones variadas: walk, run, talk_a, talk_b"
echo "   ✅ Iluminación mejorada con luz direccional"
echo "   ✅ Trayectorias más dinámicas y naturales"
echo ""
echo "🚀 Iniciando simulación..."
echo ""

cd "$WORKSPACE_DIR"

# Source workspace - navegar al directorio correcto
WORKSPACE_ROOT="${TFG_WS}"
if [ -f "$WORKSPACE_ROOT/install/setup.bash" ]; then
    source "$WORKSPACE_ROOT/install/setup.bash"
    cd "$WORKSPACE_ROOT"
fi

# Launch simulation with realistic world
ros2 launch src/tfg_workflow/launch/sim_with_people.launch.py \
    world:=warehouse_actors_realistic \
    rviz:=false \
    x:=-2.0 \
    y:=2.0 \
    yaw:=0.0

echo ""
echo "✅ Simulación iniciada con mundo mejorado"
echo ""
echo "💡 Tips para máxima calidad visual:"
echo "   - Maximiza la ventana de Gazebo"
echo "   - Usa 'View > Wireframe' para ver estructura"
echo "   - Usa 'View > Transparent' para ver a través de objetos"
echo "   - Presiona 'R' para resetear vista de cámara"
echo ""
echo "🔧 Para ejecutar person-following:"
echo "   cd $WORKSPACE_DIR"
echo "   bash scripts/run_pipeline.sh follow-launch"
echo ""

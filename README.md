# TFG — Comportamiento de seguimiento seguro para un robot móvil

Trabajo de Fin de Grado que implementa, sobre un **TurtleBot 4** en simulación (**ROS 2 Jazzy**
+ **Gazebo Harmonic**), un sistema de **detección de personas** (YOLO), **seguimiento
multi-objeto** y un **comportamiento de seguimiento seguro**: el robot fija un objetivo por
proximidad, lo sigue con seguimiento visual aunque se crucen otras personas, y evita los
obstáculos del entorno sin colisionar.

## Requisitos

- Ubuntu 24.04
- ROS 2 Jazzy y Gazebo Harmonic (con la integración `ros_gz`)
- Paquetes TurtleBot 4 (`turtlebot4_gz_bringup`, `irobot_create_*`)
- Python 3.12 y `ultralytics` (YOLO). GPU NVIDIA recomendada.

> El modelo de detección en vivo (`yolo11s.pt`, COCO) y los modelos de escenario de Gazebo
> (Fuel) se **descargan automáticamente** la primera vez; no se versionan en el repositorio.

## Compilación

```bash
cd ~/tfg_ws            # raíz del workspace (la que contiene src/)
colcon build --symlink-install
source install/setup.bash
```

## Configuración (GPU NVIDIA y W&B)

**GPU NVIDIA (importante para el láser).** Gazebo debe ejecutarse en la GPU; si no, el LiDAR
simulado (`GpuLidar`) puede fallar y devolver un valor de rango FIJO (no "ve" obstáculos ni
personas). `run_tb4_sim.sh` ya exporta las variables necesarias:

```bash
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
# En algunas máquinas (glvnd elige Mesa, aviso `libEGL ... driver (null)`) hace falta además
# forzar el vendor EGL de NVIDIA para que el GpuLidar renderice:
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

(En portátiles con gráfica híbrida conviene además dejarlas en `~/.bashrc`.) Para comprobar que
Gazebo usa la GPU, con la simulación lanzada ejecuta `nvidia-smi`: deben aparecer los procesos
`gz sim server` y `gz sim gui`. Para comprobar el láser, mira que `ros2 topic echo /scan` dé
rangos **variables** (no un valor fijo); también puedes visualizarlo con
`ros2 launch turtlebot4_viz view_model.launch.py` añadiendo el `LaserScan` en RViz.

**Weights & Biases es opcional.** El seguidor registra telemetría en W&B si hay sesión iniciada
(`wandb login`); si no la hay, arranca igual y continúa **sin telemetría** (no es necesario para
ejecutar la demo). También puede desactivarse con `use_wandb:=false`.

## Ejecución de la demo de seguimiento

En tres terminales (con el workspace *sourced* en cada una):

```bash
# 1) Simulación: mundo de almacén con personas animadas
bash src/tfg_workflow/scripts/run_tb4_sim.sh warehouse_unified

# 2) Percepción + control (detector YOLO + seguidor)
ros2 launch src/tfg_workflow/launch/person_follow.launch.py

# 3) (opcional) Visualizar las detecciones; el objetivo fijado se resalta en verde
ros2 run rqt_image_view rqt_image_view /person_detections/image
```

> Los nodos (`person_detector.py`, `person_follower.py`) se arrancan desde el *launch*
> (`person_follow.launch.py`). Tras `colcon build` quedan también instalados en
> `lib/tfg_workflow`. La distancia al objetivo se obtiene relacionando el láser con la imagen;
> si el láser no incide sobre la persona se recurre a la cámara de profundidad OAK-D.

## Estructura del repositorio

```
src/tfg_workflow/
├── scripts/        # Detección, seguimiento, evaluación y comportamiento de seguimiento
│   ├── person_detector.py     # YOLO + tracking → /person_detections (con distancia)
│   ├── person_follower.py     # controlador de seguimiento seguro → /cmd_vel
│   ├── run_tb4_sim.sh         # lanza la simulación
│   ├── test_follower_logic.py # banco de pruebas offline del controlador
│   └── ...                    # entrenamiento YOLO y evaluación de tracking (MOT)
├── launch/         # person_follow / sim_with_people / turtlebot4_spawn_nodock
├── worlds/         # warehouse_unified.sdf (escenario de la demo)
├── models/         # actores animados usados en el escenario
├── config/, python_lib/, package.xml, setup.py
└── yolo11s_100ep.pt           # modelo de detección entrenado (mAP ≈ 0.913)
```

## Componentes

- **Detector (`person_detector.py`)**: suscribe `/oakd/rgb/preview/image_raw` (+ profundidad),
  ejecuta YOLO con seguimiento multi-objeto (BoT-SORT) y publica todas las personas en
  `/person_detections` con su distancia.
- **Seguidor (`person_follower.py`)**: a partir de `/person_detections` y `/scan`, fija el
  objetivo por proximidad (~3 s), lo sigue por identificador (robusto a cruces) manteniendo la
  distancia de consigna, y aplica una capa de evitación basada en LiDAR. Publica en `/cmd_vel`.
  Máquina de estados: `SEARCHING → ACQUIRING → FOLLOWING`, con `STOPPED` de seguridad.

Parámetros principales (por defecto): `safe_dist = 0.5 m`, `target_dist = 3.0 m`,
`max_linear = 0.6 m/s`, `max_angular = 1.2 rad/s`.

## Evaluación de seguimiento multi-objeto

Los scripts de `scripts/` permiten reproducir la evaluación de trackers (BoT-SORT, StrongSORT,
ByteTrack, OC-SORT, DeepOCSORT) sobre secuencias en formato MOTChallenge con *ground-truth*
geométrico (`capture_mot_dataset.py`, `evaluate_mot_euclid.py`, `aggregate_mot_euclid.py`,
`run_all_boxmot.py`). Los conjuntos de datos son voluminosos y no se versionan; están
disponibles a petición.

## Información académica

- Universidad de Granada — E.T.S. de Ingenierías Informática y de Telecomunicación
- Curso 2025–2026
- Tutor: Eugenio Aguirre Molina

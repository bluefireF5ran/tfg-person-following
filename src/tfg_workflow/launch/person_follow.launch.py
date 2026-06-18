"""Launch person-following behavior: detector + follower nodes.

Usage (with simulation already running):
  ros2 launch src/tfg_workflow/launch/person_follow.launch.py

With custom parameters:
  ros2 launch src/tfg_workflow/launch/person_follow.launch.py \
      model:=yolo11s.pt safe_dist:=0.6 target_dist:=1.5
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")

ARGUMENTS = [
    DeclareLaunchArgument(
        "model", default_value="yolo11s.pt",
        description="YOLO model. Para el SEGUIMIENTO en vivo se usa yolo11s.pt (COCO): detecta "
                    "personas PARCIALES (la OAK-D va baja y a ~3 m corta la cabeza) mucho mejor "
                    "que yolo11s_100ep.pt, que esta afinado al dataset del almacen (eval de tracking)"),
    DeclareLaunchArgument(
        "camera_topic", default_value="/oakd/rgb/preview/image_raw",
        description="Camera Image topic"),
    DeclareLaunchArgument(
        "conf", default_value="0.3",
        description="Umbral de confianza YOLO (bajo: la persona frontal sale cortada por arriba "
                    "-> confianza menor; 0.3 la recupera sin meter falsos en esta escena)"),
    DeclareLaunchArgument(
        "tracker", default_value="botsort.yaml",
        description="Tracker MOT de Ultralytics (botsort.yaml | bytetrack.yaml)"),
    DeclareLaunchArgument(
        "depth_topic", default_value="/oakd/rgb/preview/depth",
        description="Topic de profundidad OAK-D (distancia al objetivo)"),
    DeclareLaunchArgument(
        "acquire_dwell_s", default_value="3.0",
        description="Segundos que la persona mas cercana debe mantenerse para fijarse"),
    DeclareLaunchArgument(
        "acquire_max_range", default_value="3.3",
        description="Distancia maxima (m) para fijar a una persona como objetivo (solo el objetivo a 3.0 m entra; el resto >=3.9 m)"),
    DeclareLaunchArgument(
        "avoid_dist", default_value="1.2",
        description="Distancia (m) a la que empieza el sesgo de giro de esquive (frenado progresivo desde 1.5 m)"),
    DeclareLaunchArgument(
        "safe_dist", default_value="0.5",
        description="Emergency stop distance (m)"),
    DeclareLaunchArgument(
        "target_dist", default_value="3.0",
        description="Desired following distance (m) -- camara OAK-D baja: <3 m solo ve piernas, se mantiene a 3.0 m"),
    DeclareLaunchArgument(
        "max_linear", default_value="0.6",
        description="Max forward speed (m/s)"),
    DeclareLaunchArgument(
        "max_angular", default_value="1.2",
        description="Max rotation speed (rad/s)"),
    DeclareLaunchArgument(
        "search_angular", default_value="0.5",
        description="Rotation speed while SEARCHING (rad/s) -- lento para que el objetivo quieto este ~2.4s en el FOV por barrido y de tiempo a engancharlo"),
    DeclareLaunchArgument(
        "bbox_scale", default_value="1.0",
        description="Actor scale factor for bbox distance (0.9-1.1)"),
]


def generate_launch_description():
    # Person detector node (YOLO)
    detector = Node(
        package=None,
        executable="python3",
        name="person_detector",
        output="screen",
        arguments=[
            str(Path(SCRIPTS_DIR) / "person_detector.py"),
            "--model", LaunchConfiguration("model"),
            "--topic", LaunchConfiguration("camera_topic"),
            "--depth-topic", LaunchConfiguration("depth_topic"),
            "--conf", LaunchConfiguration("conf"),
            "--tracker", LaunchConfiguration("tracker"),
        ],
    )

    # Person follower node (controller)
    follower = Node(
        package=None,
        executable="python3",
        name="person_follower",
        output="screen",
        arguments=[
            str(Path(SCRIPTS_DIR) / "person_follower.py"),
            "--safe-dist", LaunchConfiguration("safe_dist"),
            "--target-dist", LaunchConfiguration("target_dist"),
            "--max-linear", LaunchConfiguration("max_linear"),
            "--max-angular", LaunchConfiguration("max_angular"),
            "--search-angular", LaunchConfiguration("search_angular"),
            "--acquire-dwell-s", LaunchConfiguration("acquire_dwell_s"),
            "--acquire-max-range", LaunchConfiguration("acquire_max_range"),
            "--avoid-dist", LaunchConfiguration("avoid_dist"),
            "--bbox-scale", LaunchConfiguration("bbox_scale"),
        ],
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(detector)
    ld.add_action(follower)
    return ld

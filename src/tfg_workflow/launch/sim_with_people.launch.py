import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

BASE_DIR = Path(__file__).resolve().parent.parent
CUSTOM_WORLDS_DIR = str(BASE_DIR / 'worlds')
CUSTOM_MODELS_DIR = str(BASE_DIR / 'models')

ARGUMENTS = [
    DeclareLaunchArgument('namespace', default_value='',
                          description='Robot namespace'),
    DeclareLaunchArgument('rviz', default_value='false',
                          choices=['true', 'false'], description='Start rviz.'),
    DeclareLaunchArgument('world', default_value='warehouse_actors',
                          description='Simulation World'),
    DeclareLaunchArgument('model', default_value='standard',
                          choices=['standard', 'lite'],
                          description='Turtlebot4 Model'),
    DeclareLaunchArgument('headless', default_value='false',
                          choices=['true', 'false'],
                          description='Run Gazebo in headless/server mode'),
]

for pose_element in ['x', 'y', 'z', 'yaw']:
    ARGUMENTS.append(DeclareLaunchArgument(
        pose_element, default_value='0.0',
        description=f'{pose_element} component of the robot pose.'))


def generate_launch_description():
    pkg_turtlebot4_gz_bringup = get_package_share_directory(
        'turtlebot4_gz_bringup')
    pkg_turtlebot4_gz_gui_plugins = get_package_share_directory(
        'turtlebot4_gz_gui_plugins')
    pkg_turtlebot4_description = get_package_share_directory(
        'turtlebot4_description')
    pkg_irobot_create_description = get_package_share_directory(
        'irobot_create_description')
    pkg_irobot_create_gz_bringup = get_package_share_directory(
        'irobot_create_gz_bringup')
    pkg_irobot_create_gz_plugins = get_package_share_directory(
        'irobot_create_gz_plugins')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=':'.join([
            CUSTOM_MODELS_DIR,
            CUSTOM_WORLDS_DIR,
            os.path.join(pkg_turtlebot4_gz_bringup, 'worlds'),
            os.path.join(pkg_irobot_create_gz_bringup, 'worlds'),
            str(Path(pkg_turtlebot4_description).parent.resolve()),
            str(Path(pkg_irobot_create_description).parent.resolve()),
        ])
    )

    gz_gui_plugin_path = SetEnvironmentVariable(
        name='GZ_GUI_PLUGIN_PATH',
        value=':'.join([
            os.path.join(pkg_turtlebot4_gz_gui_plugins, 'lib'),
            os.path.join(pkg_irobot_create_gz_plugins, 'lib'),
        ])
    )

    fastrtps_config = SetEnvironmentVariable(
        name='FASTRTPS_DEFAULT_PROFILES_FILE',
        value=str(BASE_DIR / 'config' / 'fastrtps_no_shm.xml'),
    )

    disable_loaned_messages = SetEnvironmentVariable(
        name='ROS_DISABLE_LOANED_MESSAGES',
        value='1',
    )

    pythonpath_intercept = SetEnvironmentVariable(
        name='PYTHONPATH',
        value=f"{BASE_DIR / 'python_lib'}:{os.environ.get('PYTHONPATH', '')}",
    )

    gz_sim_launch = PathJoinSubstitution(
        [pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'])
    gz_args_str = [
        LaunchConfiguration('world'), '.sdf', ' -r', ' -v 4',
    ]
    gz_args_str += [' --gui-config ',
        PathJoinSubstitution([
            pkg_turtlebot4_gz_bringup,
            'gui',
            LaunchConfiguration('model'),
            'gui.config',
        ]),
    ]

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([gz_sim_launch]),
        launch_arguments=[('gz_args', gz_args_str)],
    )

    clock_bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        name='clock_bridge', output='screen',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
    )

    # Spawn LOCAL con el dock reubicado lejos (turtlebot4_spawn_nodock.launch.py): el spawn del
    # sistema generaba el "standard_dock" 0.157 m justo delante del robot, entre este y el
    # objetivo, y el robot lo embestia al seguir (el LiDAR no lo ve para esquivarlo).
    local_spawn_launch = str(BASE_DIR / 'launch' / 'turtlebot4_spawn_nodock.launch.py')
    robot_spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([local_spawn_launch]),
        launch_arguments=[
            ('namespace', LaunchConfiguration('namespace')),
            ('rviz', LaunchConfiguration('rviz')),
            ('x', LaunchConfiguration('x')),
            ('y', LaunchConfiguration('y')),
            ('z', LaunchConfiguration('z')),
            ('yaw', LaunchConfiguration('yaw')),
        ],
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(gz_resource_path)
    ld.add_action(gz_gui_plugin_path)
    ld.add_action(fastrtps_config)
    ld.add_action(disable_loaned_messages)
    ld.add_action(pythonpath_intercept)
    ld.add_action(gazebo)
    ld.add_action(clock_bridge)
    ld.add_action(robot_spawn)
    return ld

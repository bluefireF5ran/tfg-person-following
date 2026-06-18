#!/usr/bin/env python3
import os
import sys
import time
import tempfile
import subprocess
import rclpy
from rclpy.node import Node
from controller_manager_msgs.srv import ListControllers


class DeferredWorldPopulator(Node):
    def __init__(self):
        super().__init__('deferred_world_populator')
        self.client = self.create_client(ListControllers, '/controller_manager/list_controllers')
        self.get_logger().info('DeferredWorldPopulator inicializado. Vigilando el arranque del robot...')

    def wait_for_robot_active(self, timeout_sec=120.0):
        self.get_logger().info('Esperando a que el controller_manager del TurtleBot 4 esté disponible...')
        if not self.client.wait_for_service(timeout_sec=30.0):
            self.get_logger().error('El servicio /controller_manager/list_controllers no está disponible.')
            return False

        t0 = time.monotonic()
        req = ListControllers.Request()

        while time.monotonic() - t0 < timeout_sec:
            future = self.client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

            if future.done() and future.result() is not None:
                controllers = future.result().controller
                states = {c.name: c.state for c in controllers}

                jsb_active = states.get('joint_state_broadcaster') == 'active'
                diff_active = states.get('diffdrive_controller') == 'active'

                if jsb_active and diff_active:
                    self.get_logger().info('¡Robot detectado en estado ACTIVE! Procediendo al poblado diferido...')
                    return True
                else:
                    self.get_logger().info(
                        f'Esperando activacion de controladores... '
                        f'(joint_state_broadcaster: {states.get("joint_state_broadcaster")}, '
                        f'diffdrive_controller: {states.get("diffdrive_controller")})'
                    )
            time.sleep(2.0)

        self.get_logger().error('Timeout esperando a que el robot estuviera activo.')
        return False

    def populate(self):
        actores = ['walk.dae', 'run.dae', 'talk_a.dae', 'talk_b.dae']
        poses = [
            "-5 5.5 1.0 0 0 0",
            "-4 3.5 1.1 0 0 0",
            "1 4.5 1.0 0 0 1.57",
            "2 5.5 0.95 0 0 -1.57",
        ]

        time.sleep(3.0)

        for i, (actor, pose) in enumerate(zip(actores, poses)):
            name = f"actor_{i}"
            self.get_logger().info(f'Preparando inyeccion de [{name}] ({actor})...')

            sdf_string = f"""<?xml version="1.0" ?>
<sdf version="1.7">
  <actor name="{name}">
    <skin>
      <filename>https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/{actor}</filename>
      <scale>1.0</scale>
    </skin>
    <animation name="action">
      <filename>https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/{actor}</filename>
      <interpolate_x>true</interpolate_x>
    </animation>
    <script>
      <loop>true</loop>
      <auto_start>true</auto_start>
      <trajectory id="0" type="action">
        <waypoint><time>0</time><pose>{pose}</pose></waypoint>
      </trajectory>
    </script>
  </actor>
</sdf>
"""

            with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as temp_file:
                temp_file.write(sdf_string)
                temp_path = temp_file.name

            cmd = [
                'ros2', 'run', 'ros_gz_sim', 'create',
                '-world', 'warehouse_unified',
                '-file', temp_path,
                '-name', name,
            ]

            try:
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                self.get_logger().info(f'¡[{name}] inyectado correctamente en el almacen!')
            except subprocess.CalledProcessError as e:
                self.get_logger().error(f'Error al inyectar [{name}]: {e.stderr.decode()}')
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        self.get_logger().info('¡Fase de poblado diferido completada con exito!')


def main():
    rclpy.init()
    node = DeferredWorldPopulator()
    if node.wait_for_robot_active():
        node.populate()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

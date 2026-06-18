#!/usr/bin/env python3
import sys
import time
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from controller_manager_msgs.srv import (
    ListControllers, LoadController, ConfigureController, SwitchController
)


class PatientSpawner(Node):
    def __init__(self):
        super().__init__('patient_spawner')
        self.controllers = ['joint_state_broadcaster', 'diffdrive_controller']

        self.client_list = self.create_client(ListControllers, '/controller_manager/list_controllers')
        self.client_load = self.create_client(LoadController, '/controller_manager/load_controller')
        self.client_configure = self.create_client(ConfigureController, '/controller_manager/configure_controller')
        self.client_switch = self.create_client(SwitchController, '/controller_manager/switch_controller')

    def _call_service_with_retry(self, client, request, timeout=10.0, max_attempts=5):
        for attempt in range(max_attempts):
            if not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warn(f'Servicio {client.srv_name} no disponible, reintentando descubrimiento...')
                continue

            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)

            if future.done():
                return future.result()

            self.get_logger().warn(
                f'La llamada a {client.srv_name} expiró (Timeout {timeout}s). '
                f'Intento {attempt+1}/{max_attempts}'
            )
        return None

    def wait_for_cm(self):
        self.get_logger().info('Esperando disponibilidad del controller_manager (Límite: 300s)...')
        t0 = time.monotonic()
        while time.monotonic() - t0 < 300.0:
            res = self._call_service_with_retry(self.client_list, ListControllers.Request(), timeout=5.0, max_attempts=1)
            if res is not None:
                self.get_logger().info(f'¡controller_manager detectado y operativo! ({time.monotonic()-t0:.1f}s)')
                return True
            time.sleep(2.0)
        return False

    def get_state(self, name):
        res = self._call_service_with_retry(self.client_list, ListControllers.Request(), timeout=5.0, max_attempts=3)
        if res:
            for c in res.controller:
                if c.name == name:
                    return c.state
        return None

    def run(self):
        if not self.wait_for_cm():
            self.get_logger().error('Error crítico: El nodo controller_manager nunca apareció.')
            return 1

        for ctrl in self.controllers:
            state = self.get_state(ctrl)
            self.get_logger().info(f'[{ctrl}] estado detectado = {state}')

            if state == 'active':
                self.get_logger().info(f'[{ctrl}] ya se encuentra activo en la simulación.')
                continue

            if state is None:
                self.get_logger().info(f'[{ctrl}] cargando de forma resiliente...')
                req = LoadController.Request(name=ctrl)
                res = self._call_service_with_retry(self.client_load, req, timeout=20.0, max_attempts=5)
                if not (res and res.ok):
                    state = self.get_state(ctrl)
                    if state is None:
                        self.get_logger().error(f'[{ctrl}] fallo catastrófico en la carga.')
                        continue
                state = 'unconfigured'

            if state == 'unconfigured':
                self.get_logger().info(f'[{ctrl}] configurando de forma resiliente...')
                t0 = time.monotonic()
                req = ConfigureController.Request(name=ctrl)
                res = self._call_service_with_retry(self.client_configure, req, timeout=20.0, max_attempts=5)
                elapsed = time.monotonic() - t0
                if res and res.ok:
                    self.get_logger().info(f'[{ctrl}] configurado ({elapsed:.1f}s)')
                    state = 'inactive'
                else:
                    state = self.get_state(ctrl)
                    if state in ['inactive', 'active']:
                        self.get_logger().info(f'[{ctrl}] configurado (confirmado de forma indirecta)')
                        state = 'inactive'
                    else:
                        self.get_logger().error(f'[{ctrl}] fallo catastrófico al configurar.')
                        continue

            if state == 'inactive':
                self.get_logger().info(f'[{ctrl}] activando de forma resiliente...')
                req = SwitchController.Request()
                req.activate_controllers = [ctrl]
                req.strictness = SwitchController.Request.BEST_EFFORT
                req.timeout = Duration(sec=15, nanosec=0)

                res = self._call_service_with_retry(self.client_switch, req, timeout=20.0, max_attempts=5)
                if res and res.ok:
                    self.get_logger().info(f'[{ctrl}] ¡activado de forma exitosa!')
                else:
                    if self.get_state(ctrl) == 'active':
                        self.get_logger().info(f'[{ctrl}] ya se encuentra activo')
                    else:
                        self.get_logger().error(f'[{ctrl}] fallo de activación.')

        self.get_logger().info('=== PatientSpawner finalizado ===')
        final_jsb = self.get_state('joint_state_broadcaster')
        final_dc = self.get_state('diffdrive_controller')
        self.get_logger().info(f'Estado Final en Red: JSB={final_jsb}, DiffDrive={final_dc}')
        return 0


def main():
    rclpy.init()
    s = PatientSpawner()
    rc = s.run()
    s.destroy_node()
    rclpy.shutdown()
    sys.exit(rc)


if __name__ == '__main__':
    main()

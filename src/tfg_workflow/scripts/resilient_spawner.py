#!/usr/bin/env python3
# Copyright (c) 2026 TFG Workflow
# Licensed under the Apache License, Version 2.0

import sys
import rclpy
from rclpy.node import Node
from controller_manager_msgs.srv import (
    ListControllers, LoadController, ConfigureController, SwitchController
)

class SmartResilientSpawner(Node):
    def __init__(self):
        super().__init__('smart_resilient_spawner')
        self.controllers = ['joint_state_broadcaster', 'diffdrive_controller']

    def _call_service(self, srv_type, srv_name, request, timeout=5.0):
        client = self.create_client(srv_type, srv_name)
        if not client.wait_for_service(timeout_sec=2.0):
            return None
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.result() if future.done() else None

    def get_controller_state(self, name, max_attempts=3):
        for _ in range(max_attempts):
            res = self._call_service(ListControllers, '/controller_manager/list_controllers', ListControllers.Request(), timeout=3.0)
            if res:
                for c in res.controller:
                    if c.name == name:
                        return c.state
        return None

    def wait_for_controller_manager(self, max_attempts=15, attempt_timeout=4.0):
        self.get_logger().info('Buscando controller_manager en la red DDS...')
        for attempt in range(max_attempts):
            res = self._call_service(ListControllers, '/controller_manager/list_controllers', ListControllers.Request(), timeout=attempt_timeout)
            if res is not None:
                self.get_logger().info('¡Conexión establecida con el controller_manager!')
                return True
        return False

    def run(self):
        if not self.wait_for_controller_manager():
            self.get_logger().error('Fallo crítico: El nodo controller_manager no responde.')
            return 1

        for ctrl in self.controllers:
            state = self.get_controller_state(ctrl)
            self.get_logger().info(f'Estado detectado para [{ctrl}]: {state}')
            
            if state == 'active':
                self.get_logger().info(f'[{ctrl}] ya se encuentra activo. Saltando de forma segura.')
                continue
                
            if state is None:
                self.get_logger().info(f'[{ctrl}] no está cargado. Cargando...')
                if not self._load_controller(ctrl):
                    self.get_logger().error(f'Fallo al cargar [{ctrl}]')
                    continue
                state = 'unconfigured'
                
            if state == 'unconfigured':
                self.get_logger().info(f'[{ctrl}] está unconfigured. Configurando...')
                if not self._configure_controller(ctrl):
                    self.get_logger().error(f'Fallo al configurar [{ctrl}]')
                    continue
                state = 'inactive'
                
            if state == 'inactive':
                self.get_logger().info(f'[{ctrl}] está inactivo. Activando...')
                if not self._activate_controller(ctrl):
                    self.get_logger().error(f'Fallo al activar [{ctrl}]')
                    continue
                self.get_logger().info(f'¡[{ctrl}] configurado y activado con éxito de forma resiliente!')
                
        return 0

    def _load_controller(self, name, max_attempts=5, attempt_timeout=4.0):
        req = LoadController.Request(name=name)
        for attempt in range(max_attempts):
            res = self._call_service(LoadController, '/controller_manager/load_controller', req, timeout=attempt_timeout)
            if res and res.ok:
                return True
            if self.get_controller_state(name) is not None:
                return True
        return False

    def _configure_controller(self, name, max_attempts=5, attempt_timeout=4.0):
        req = ConfigureController.Request(name=name)
        for attempt in range(max_attempts):
            res = self._call_service(ConfigureController, '/controller_manager/configure_controller', req, timeout=attempt_timeout)
            if res and res.ok:
                return True
            if self.get_controller_state(name) in ['inactive', 'active']:
                return True
        return False

    def _activate_controller(self, name, max_attempts=5, attempt_timeout=4.0):
        req = SwitchController.Request()
        req.activate_controllers = [name]
        req.strictness = SwitchController.Request.BEST_EFFORT
        for attempt in range(max_attempts):
            res = self._call_service(SwitchController, '/controller_manager/switch_controller', req, timeout=attempt_timeout)
            if res and res.ok:
                return True
            if self.get_controller_state(name) == 'active':
                return True
        return False

def main():
    rclpy.init()
    spawner = SmartResilientSpawner()
    rc = spawner.run()
    spawner.destroy_node()
    rclpy.shutdown()
    sys.exit(rc)

if __name__ == '__main__':
    main()

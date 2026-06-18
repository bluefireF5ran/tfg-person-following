#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from controller_manager_msgs.srv import LoadController, SwitchController, ListControllers

SVC_TIMEOUT = 60.0


class SpawnControllers(Node):
    def __init__(self):
        super().__init__('spawn_controllers')

    def _service_call(self, srv_type, srv_name, req, timeout=SVC_TIMEOUT):
        cli = self.create_client(srv_type, srv_name)
        while not cli.wait_for_service(timeout_sec=timeout):
            self.get_logger().warn(f'{srv_name} not available, retrying...')
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if future.done():
            return future.result()
        self.get_logger().warn(f'{srv_name} timed out ({timeout}s)')
        return None

    def _is_loaded(self, name):
        result = self._service_call(ListControllers, '/controller_manager/list_controllers',
                                    ListControllers.Request())
        if not result:
            return False
        for c in result.controller:
            if c.name == name:
                return True
        return False

    def _load(self, name):
        if self._is_loaded(name):
            self.get_logger().info(f'{name} already loaded')
            return True
        req = LoadController.Request()
        req.name = name
        for attempt in range(3):
            result = self._service_call(LoadController, '/controller_manager/load_controller', req)
            if result and result.ok:
                self.get_logger().info(f'{name} loaded')
                return True
            if self._is_loaded(name):
                self.get_logger().info(f'{name} loaded (by another process)')
                return True
            self.get_logger().warn(f'{name} load attempt {attempt+1} failed')
        self.get_logger().error(f'{name} NOT loaded after 3 attempts')
        return False

    def _activate(self, name):
        req = SwitchController.Request()
        req.activate_controllers = [name]
        req.strictness = SwitchController.Request.BEST_EFFORT
        result = self._service_call(SwitchController, '/controller_manager/switch_controller', req)
        if result and result.ok:
            self.get_logger().info(f'{name} activated')
            return True
        self.get_logger().error(f'{name} activation failed')
        return False

    def run(self):
        if not self._service_call(ListControllers, '/controller_manager/list_controllers',
                                  ListControllers.Request(), timeout=SVC_TIMEOUT):
            self.get_logger().error('controller_manager not available')
            return 1
        self.get_logger().info('controller_manager available')

        for ctrl in ['joint_state_broadcaster', 'diffdrive_controller']:
            if self._load(ctrl):
                self._activate(ctrl)
        return 0


def main():
    rclpy.init()
    spawner = SpawnControllers()
    rc = spawner.run()
    spawner.destroy_node()
    rclpy.shutdown()
    sys.exit(rc)


if __name__ == '__main__':
    main()

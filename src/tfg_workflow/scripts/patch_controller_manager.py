#!/usr/bin/env python3
import os
import shutil
import sys

ROS_PACKAGE_PATH = '/opt/ros/jazzy/lib/python3.12/site-packages/controller_manager'
WORKSPACE_LIB_PATH = os.path.expanduser('~/tfg_ws/src/tfg_workflow/python_lib')
TARGET_PACKAGE_PATH = os.path.join(WORKSPACE_LIB_PATH, 'controller_manager')


def apply_patch():
    print('[INFO] Iniciando el parcheo definitivo de controller_manager...')

    if not os.path.exists(ROS_PACKAGE_PATH):
        print(f'[ERROR] No se detecto la instalacion de ROS 2 en: {ROS_PACKAGE_PATH}')
        sys.exit(1)

    if os.path.exists(TARGET_PACKAGE_PATH):
        shutil.rmtree(TARGET_PACKAGE_PATH)

    os.makedirs(WORKSPACE_LIB_PATH, exist_ok=True)

    print('[INFO] Copiando el paquete original de ROS 2 al espacio de usuario...')
    shutil.copytree(ROS_PACKAGE_PATH, TARGET_PACKAGE_PATH)

    services_file = os.path.join(TARGET_PACKAGE_PATH, 'controller_manager_services.py')
    if os.path.exists(services_file):
        with open(services_file, 'r') as f:
            content = f.read()
        modified = content.replace('call_timeout=10.0', 'call_timeout=120.0')
        with open(services_file, 'w') as f:
            f.write(modified)
        print('[INFO] Parche aplicado en controller_manager_services.py (default=120s).')

    spawner_file = os.path.join(TARGET_PACKAGE_PATH, 'spawner.py')
    if os.path.exists(spawner_file):
        with open(spawner_file, 'r') as f:
            spawner_content = f.read()
        spawner_modified = spawner_content.replace('default=10.0', 'default=120.0')
        with open(spawner_file, 'w') as f:
            f.write(spawner_modified)
        print('[INFO] Parche aplicado en el analizador de argumentos de spawner.py (default=120s).')

    print('[SUCCESS] Entorno blindado localmente con exito!')


if __name__ == '__main__':
    apply_patch()

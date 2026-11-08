from setuptools import setup
from glob import glob
import os

package_name = 'tfg_workflow'

def collect_files(source_dir, install_dir):
    """Colecciona de forma recursiva todos los subdirectorios preservando la estructura para instalación."""
    entries = []
    for root, _, filenames in os.walk(source_dir):
        if not filenames:
            continue
        rel = os.path.relpath(root, source_dir)
        target = os.path.join(install_dir, rel) if rel != '.' else install_dir
        entries.append((target, [os.path.join(root, f) for f in filenames]))
    return entries

setup(
    name=package_name,
    version='0.1.0',
    packages=[],
    data_files=[
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        *collect_files('models', os.path.join('share', package_name, 'models')),
        *collect_files('python_lib', os.path.join('share', package_name, 'python_lib')),
        # Registro crítico: Instalamos los scripts en lib/ para que ros2 run los localice al instante
        (os.path.join('lib', package_name), [
            'scripts/person_detector.py',
            'scripts/person_follower.py',
            'scripts/download_local_actors.py',
            'scripts/patch_controller_manager.py',
            'scripts/analyze_telemetry.py',
            'scripts/benchmark_fps_yolo.py',
            'scripts/track_persons.py',
            'scripts/evaluate_mot.py',
            'scripts/capture_mot_dataset.py',
            'scripts/calibrate_actor_boxes.py',
            'scripts/track_boxmot.py',
            'scripts/run_all_boxmot.py',
            'scripts/aggregate_mot.py',
            'scripts/evaluate_mot_euclid.py',
            'scripts/coco_to_mot.py',
            'scripts/robot7_make_gt.py',
            'scripts/aggregate_robot7.py',
            'scripts/aggregate_mot_euclid.py',
            'scripts/tune_refine.py',
            'scripts/recompute_gt_offline.py',
        ]),
    ],
    scripts=[
        'scripts/person_detector.py',
        'scripts/person_follower.py',
        'scripts/download_local_actors.py',
        'scripts/patch_controller_manager.py',
        'scripts/analyze_telemetry.py',
        'scripts/benchmark_fps_yolo.py',
        'scripts/track_persons.py',
        'scripts/evaluate_mot.py',
        'scripts/capture_mot_dataset.py',
        'scripts/calibrate_actor_boxes.py',
        'scripts/track_boxmot.py',
        'scripts/run_all_boxmot.py',
        'scripts/aggregate_mot.py',
        'scripts/evaluate_mot_euclid.py',
        'scripts/coco_to_mot.py',
        'scripts/robot7_make_gt.py',
        'scripts/aggregate_robot7.py',
        'scripts/aggregate_mot_euclid.py',
        'scripts/tune_refine.py',
        'scripts/recompute_gt_offline.py',
        'scripts/run_tb4_sim.sh',
        'scripts/generate_mot_dataset.sh',
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Francisco Merchán Cordero',
    maintainer_email='pacomerchanc@gmail.com',
    description='TFG workflow package',
    license='MIT',
)

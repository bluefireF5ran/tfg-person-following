#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys

WORKSPACE_MODELS_DIR = os.path.expanduser('~/tfg_ws/src/tfg_workflow/models')
FUEL_CACHE = os.path.expanduser('~/.gz/fuel/fuel.gazebosim.org')

MODELS_TO_DOWNLOAD = [
    {
        'owner': 'Luca',
        'model': 'FemaleVisitorWalk',
        'local_name': 'actor_female_visitor',
        'desc': 'Visitante Femenina Casual (Luca)'
    },
    {
        'owner': 'Luca',
        'model': 'NurseFemaleWalk',
        'local_name': 'actor_nurse',
        'desc': 'Enfermera Femenina (Luca)'
    },
    {
        'owner': 'OpenRobotics',
        'model': 'Male visitor',
        'local_name': 'actor_male_visitor',
        'desc': 'Visitante Masculino (OpenRobotics)'
    },
]


def find_fuel_cache_dir(owner, model_name):
    """Find the cached model directory, case-insensitive."""
    owner_dir = os.path.join(FUEL_CACHE, owner.lower())
    if not os.path.isdir(owner_dir):
        owner_dir = os.path.join(FUEL_CACHE, owner)
    if not os.path.isdir(owner_dir):
        return None
    models_dir = os.path.join(owner_dir, 'models')
    if not os.path.isdir(models_dir):
        return None
    for entry in os.listdir(models_dir):
        if entry.lower() == model_name.lower():
            return os.path.join(models_dir, entry)
    return None


def download_via_gz(url):
    """Use gz fuel download to download a model."""
    try:
        result = subprocess.run(
            ['gz', 'fuel', 'download', '-u', url],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode == 0:
            return True
        print(f'  stderr: {result.stderr[:200]}')
        return False
    except subprocess.TimeoutExpired:
        print('  Timeout downloading model')
        return False
    except FileNotFoundError:
        print('  gz command not found')
        return False


def copy_to_local(owner, model_name, local_name):
    """Copy from Fuel cache to workspace models directory."""
    src_dir = find_fuel_cache_dir(owner, model_name)
    if not src_dir:
        # Try to find the latest version directory
        src_dir = find_fuel_cache_dir(owner, model_name)
        if not src_dir:
            # Search broadly
            for root, dirs, files in os.walk(FUEL_CACHE):
                for d in dirs:
                    if d.lower() == model_name.lower():
                        src_dir = os.path.join(root, d)
                        break
                if src_dir:
                    break

    if not src_dir:
        return False

    # Find the highest version number
    versions = []
    for entry in os.listdir(src_dir):
        if entry.isdigit():
            versions.append(int(entry))
    if not versions:
        return False
    latest_ver = str(max(versions))
    src_ver_dir = os.path.join(src_dir, latest_ver)

    dst_dir = os.path.join(WORKSPACE_MODELS_DIR, local_name)
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)

    shutil.copytree(src_ver_dir, dst_dir)
    return True


def main():
    print('=== Descargando modelos de actores locales ===')
    os.makedirs(WORKSPACE_MODELS_DIR, exist_ok=True)

    for model in MODELS_TO_DOWNLOAD:
        local_dir = os.path.join(WORKSPACE_MODELS_DIR, model['local_name'])
        if os.path.exists(local_dir):
            print(f'[OK] {model["local_name"]} ya existe localmente')
            continue

        url = f'https://fuel.gazebosim.org/1.0/{model["owner"]}/models/{model["model"]}'
        print(f'[INFO] Descargando {model["local_name"]} ({model["desc"]})...')
        print(f'      URL: {url}')

        ok = download_via_gz(url)
        if not ok:
            print(f'[ERROR] Fallo descarga {model["local_name"]}')
            continue

        ok = copy_to_local(model['owner'], model['model'], model['local_name'])
        if ok:
            print(f'[SUCCESS] {model["local_name"]} instalado en {local_dir}')
        else:
            print(f'[ERROR] No se encontro en cache tras descarga')

    print('=== Hecho ===')
    # Listar modelos locales
    print('\nModelos locales disponibles:')
    for entry in sorted(os.listdir(WORKSPACE_MODELS_DIR)):
        path = os.path.join(WORKSPACE_MODELS_DIR, entry)
        if os.path.isdir(path):
            size = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(path) for f in fn)
            print(f'  {entry}/  ({size//1024} KB)')


if __name__ == '__main__':
    main()

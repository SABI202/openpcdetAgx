"""
Patches nuscenes-devkit's splits.py to add a v1.0-carla split
(carla_train / carla_val), and updates create_splits_scenes() /
create_splits_logs() to expose and validate it.

Safe to re-run: checks for existing patch markers before editing.

Usage:
    python patch_nuscenes_splits.py
    # or, if nuscenes-devkit is installed somewhere non-standard:
    python patch_nuscenes_splits.py /path/to/nuscenes/utils/splits.py
"""
import sys
from pathlib import Path


CARLA_SPLITS_BLOCK = '''
# --- CARLA custom split (v1.0-carla) ---
# Scenes were renamed to be globally unique (e.g. "Town01_scene-0-1")
# via rename_scenes.py before these lists were written, since CARLA
# scene names collide across different towns/logs otherwise.
carla_train = \\
    ['Town01_scene-0-1', 'Town01_scene-0-2', 'Town01_scene-0-3', 'Town01_scene-0-4', 'Town01_scene-0-5',
     'Town01_scene-1-1', 'Town01_scene-1-2', 'Town01_scene-1-3', 'Town01_scene-1-4', 'Town01_scene-1-5',
     'Town01_scene-2-1', 'Town01_scene-2-2', 'Town01_scene-2-3', 'Town01_scene-2-4', 'Town01_scene-2-5',
     'Town05_scene-0-1', 'Town05_scene-0-2', 'Town05_scene-0-3', 'Town05_scene-0-4', 'Town05_scene-0-5',
     'Town05_scene-1-1', 'Town05_scene-1-2', 'Town05_scene-1-3', 'Town05_scene-1-4', 'Town05_scene-1-5',
     'Town05_scene-2-1', 'Town05_scene-2-2', 'Town05_scene-2-3', 'Town05_scene-2-4', 'Town05_scene-2-5',
     'Town05_scene-3-1', 'Town05_scene-3-2', 'Town05_scene-3-3', 'Town05_scene-3-4', 'Town05_scene-3-5',
     'Town03_scene-0-1', 'Town03_scene-0-2', 'Town03_scene-0-3', 'Town03_scene-0-4', 'Town03_scene-0-5']

carla_val = \\
    ['Town10HD_opt_scene-0-1', 'Town10HD_opt_scene-0-2', 'Town10HD_opt_scene-0-3',
     'Town10HD_opt_scene-0-4', 'Town10HD_opt_scene-0-5',
     'Town10HD_opt_scene-1-1', 'Town10HD_opt_scene-1-2', 'Town10HD_opt_scene-1-3',
     'Town10HD_opt_scene-1-4', 'Town10HD_opt_scene-1-5']
# --- end CARLA custom split ---
'''

PATCH_MARKER = "carla_train = "


def patch_splits_dict(content: str) -> str:
    old = (
        "    scene_splits = {'train': train, 'val': val, 'test': test,\n"
        "                    'mini_train': mini_train, 'mini_val': mini_val,\n"
        "                    'train_detect': train_detect, 'train_track': train_track}"
    )
    new = (
        "    scene_splits = {'train': train, 'val': val, 'test': test,\n"
        "                    'mini_train': mini_train, 'mini_val': mini_val,\n"
        "                    'train_detect': train_detect, 'train_track': train_track,\n"
        "                    'carla_train': carla_train, 'carla_val': carla_val}"
    )
    if old not in content:
        raise RuntimeError(
            "Could not find scene_splits dict to patch — "
            "nuscenes-devkit version may differ from expected."
        )
    return content.replace(old, new)


def patch_version_guard(content: str) -> str:
    old = (
        "    elif split == 'test':\n"
        "        assert version.endswith('test'), \\\n"
        "            'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)\n"
        "    else:"
    )
    new = (
        "    elif split == 'test':\n"
        "        assert version.endswith('test'), \\\n"
        "            'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)\n"
        "    elif split in {'carla_train', 'carla_val'}:\n"
        "        assert version == 'v1.0-carla', \\\n"
        "            'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)\n"
        "    else:"
    )
    if old not in content:
        raise RuntimeError(
            "Could not find version guard chain to patch — "
            "nuscenes-devkit version may differ from expected."
        )
    return content.replace(old, new)


def main(path_str: str = None):
    if path_str:
        path = Path(path_str)
    else:
        import nuscenes.utils.splits as _splits_mod
        path = Path(_splits_mod.__file__)

    print(f"Target file: {path}")
    content = path.read_text()

    if PATCH_MARKER in content:
        print("Already patched (carla_train found) — nothing to do.")
        return

    content = content.rstrip() + "\n" + CARLA_SPLITS_BLOCK
    content = patch_splits_dict(content)
    content = patch_version_guard(content)

    path.write_text(content)
    print("Patched successfully.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

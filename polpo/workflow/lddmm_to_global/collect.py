from polpo.dataset import Dataset, NestedDataset
from polpo.surface_mesh.deformetrica.core import (
    DeterministicAtlasResult,
    Point,
    RegistrationResult,
    ShootResult,
    TransportResult,
)


def collect_dataset(dir_config, dataset_keys):
    meshes = {}
    for outer_key, inner_keys in dataset_keys.items():
        meshes[outer_key] = {
            inner_key: Point(
                id_=f"{outer_key}-{inner_key}",
                dirname=dir_config.meshes_dir,
            )
            for inner_key in inner_keys
        }

    return NestedDataset(meshes)


def collect_local_registrations(dir_config, dataset_keys):
    dirs = {}
    for outer_key, inner_keys in dataset_keys.items():
        dirs[outer_key] = {
            inner_key: RegistrationResult.load(
                f"{outer_key}_to_{outer_key}-{inner_key}", dir_config
            )
            for inner_key in inner_keys
        }

    return NestedDataset(dirs)


def collect_global_shoots(dir_config, dataset_keys, atlas_id="gl", pole_ladder=False):
    dirs = {}
    for outer_key, inner_keys in dataset_keys.items():
        dirs[outer_key] = {
            inner_key: ShootResult.load(
                f"{atlas_id}_shoot_{outer_key}_to_{outer_key}-{inner_key}_along_{outer_key}_to_{atlas_id}",
                dir_config,
            )
            for inner_key in inner_keys
        }

    return NestedDataset(dirs)


def collect_atlases(dir_config, dataset_keys):
    return Dataset(
        {key: DeterministicAtlasResult.load(key, dir_config) for key in dataset_keys}
    )


def get_global_atlas(dir_config, atlas_id="gl"):
    return DeterministicAtlasResult.load(atlas_id, dir_config)


def collect_transports(dir_config, dataset_keys, atlas_id="gl"):
    dirs = {}

    for outer_key, inner_keys in dataset_keys.items():
        dirs[outer_key] = {
            inner_key: TransportResult.load(
                f"{outer_key}_to_{outer_key}-{inner_key}_along_{outer_key}_to_{atlas_id}",
                dir_config,
            )
            for inner_key in inner_keys
        }

    return NestedDataset(dirs)


def collect_registrations_to_global_atlas(dir_config, outer_keys, atlas_id="gl"):
    dirs = {}
    for outer_key in outer_keys:
        dirs[outer_key] = RegistrationResult.load(
            f"{outer_key}_to_{atlas_id}", dir_config
        )

    return Dataset(dirs)

from polpo.dataset import Dataset, NestedDataset
from polpo.surface_mesh.deformetrica.core import (
    DeterministicAtlasResult,
    Point,
    RegistrationResult,
    ShootResult,
    TransportResult,
)


def collect_dataset(meshes_dir, dataset_keys):
    meshes = {}
    for outer_key, inner_keys in dataset_keys.items():
        meshes[outer_key] = {
            inner_key: Point(
                id_=f"{outer_key}-{inner_key}",
                dirname=meshes_dir,
            )
            for inner_key in inner_keys
        }

    return NestedDataset(meshes)


def collect_local_registrations(registration_dir, dataset_keys):
    dirs = {}
    for outer_key, inner_keys in dataset_keys.items():
        dirs[outer_key] = {
            inner_key: RegistrationResult.from_dirname(
                registration_dir / f"{outer_key}_to_{outer_key}-{inner_key}"
            )
            for inner_key in inner_keys
        }

    return NestedDataset(dirs)


def collect_global_shoots(shoot_dir, dataset_keys, atlas_id="gl", pole_ladder=False):
    dirs = {}
    pt_str = "pole" if pole_ladder else "fan"
    for outer_key, inner_keys in dataset_keys.items():
        dirs[outer_key] = {
            inner_key: ShootResult.from_dirname(
                shoot_dir
                / f"{atlas_id}_shoot_{outer_key}_to_{outer_key}-{inner_key}_along_{pt_str}_{outer_key}_to_{atlas_id}"
            )
            for inner_key in inner_keys
        }

    return NestedDataset(dirs)


def collect_atlases(atlas_dir, dataset_keys):
    return Dataset(
        {
            key: DeterministicAtlasResult.from_dirname(atlas_dir / key)
            for key in dataset_keys
        }
    )


def get_global_atlas(atlas_dir, atlas_id="gl"):
    return DeterministicAtlasResult.from_dirname(atlas_dir / atlas_id)


def collect_transports(transport_dir, dataset_keys, atlas_id="gl", pole_ladder=False):
    dirs = {}

    pt_str = "pole" if pole_ladder else "fan"
    for outer_key, inner_keys in dataset_keys.items():
        dirs[outer_key] = {
            inner_key: TransportResult.from_dirname(
                transport_dir
                / f"{outer_key}_to_{outer_key}-{inner_key}_along_{pt_str}_{outer_key}_to_{atlas_id}"
            )
            for inner_key in inner_keys
        }

    return NestedDataset(dirs)

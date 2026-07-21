import json

from polpo.dataset import Dataset, NestedDataset
from polpo.distmat import PairwiseDistances
from polpo.mesh.deformetrica.config import DirConfig
from polpo.mesh.deformetrica.repr import (
    DeterministicAtlasDir,
    Point,
    RegistrationDir,
    ShootDir,
    TransportDir,
)
from polpo.mesh.varifold import VarifoldMetric
from polpo.utils import NestedKeyCodec
from polpo.utils.np import pairwise_dists, save_indexed_array


def varifold_metric_from_results(data, backend="auto"):
    sigma = data["kernel_tuning"]["sigma_var"]
    return VarifoldMetric(sigma=sigma, backend=backend)


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
            inner_key: RegistrationDir.from_dirname(
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
            inner_key: ShootDir.from_dirname(
                shoot_dir
                / f"{atlas_id}_shoot_{outer_key}_to_{outer_key}-{inner_key}_along_{pt_str}_{outer_key}_to_{atlas_id}"
            )
            for inner_key in inner_keys
        }

    return NestedDataset(dirs)


def collect_atlases(atlas_dir, dataset_keys):
    return Dataset(
        {
            key: DeterministicAtlasDir.from_dirname(atlas_dir / key)
            for key in dataset_keys
        }
    )


def get_global_atlas(atlas_dir, atlas_id="gl"):
    return DeterministicAtlasDir.from_dirname(atlas_dir / atlas_id)


def collect_transports(transport_dir, dataset_keys, atlas_id="gl", pole_ladder=False):
    dirs = {}

    pt_str = "pole" if pole_ladder else "fan"
    for outer_key, inner_keys in dataset_keys.items():
        dirs[outer_key] = {
            inner_key: TransportDir.from_dirname(
                transport_dir
                / f"{outer_key}_to_{outer_key}-{inner_key}_along_{pt_str}_{outer_key}_to_{atlas_id}"
            )
            for inner_key in inner_keys
        }

    return NestedDataset(dirs)


def reconstruction_error(registration_dir, dist_fnc):
    return dist_fnc(
        registration_dir.point.as_pv_surface(),
        registration_dir.reconstructed().as_pv_surface(),
    )


def atlas_reconstruction_error(atlas_dir, dist_fnc):
    return {
        point.id: dist_fnc(point.as_pv_surface(), cmp_point.as_pv_surface())
        for point, cmp_point in zip(atlas_dir.points, atlas_dir.reconstructed())
    }


def parallel_transport_dir_error(transport_dir, atlas, dist_fnc):
    return dist_fnc(
        transport_dir.reconstructed().as_pv_surface(),
        atlas,
    )


def pairwise_dist(dataset, dist_fnc):
    meshes = dataset.map_values(lambda x: x.as_pv_surface())
    flat = meshes.flatten()
    return PairwiseDistances(
        flat.keys_list(),
        pairwise_dists(flat.values_list(), dist_fnc, as_matrix=False),
    )


def local_pairwise_dist(registration_dirs, dist_fnc):
    local_rec_meshes = registration_dirs.map_values(
        lambda x: x.reconstructed().as_pv_surface()
    )
    flat = local_rec_meshes.flatten()
    return PairwiseDistances(
        flat.keys_list(),
        pairwise_dists(flat.values_list(), dist_fnc, as_matrix=False),
    )


def global_pairwise_dist(shoot_dir, dist_fnc):
    global_meshes = shoot_dir.map_values(
        lambda x: x.point().as_pv_surface(),
    )
    flat = global_meshes.flatten()
    return PairwiseDistances(
        flat.keys_list(),
        pairwise_dists(flat.values_list(), dist_fnc, as_matrix=False),
    )


def post_dists(
    outputs_dir,
    dists_folder="post_dists",
    backend="auto",
    include_local_rec_error=True,
    include_atlas_rec_error=True,
    include_global_atlas_rec_error=True,
    include_transport_error=True,
    include_local_pairwise=True,
    include_rec_local_pairwise=True,
    include_global_pairwise=True,
):
    if isinstance(dists_folder, str):
        dists_folder = outputs_dir / dists_folder

    dists_folder.mkdir(parents=True, exist_ok=True)

    # load experiment data
    with open(outputs_dir / "params.json", "r") as file:
        params = json.load(file)

    with open(outputs_dir / "results.json", "r") as file:
        results = json.load(file)

    dir_config = DirConfig(
        outputs_dir=outputs_dir,
        **{key: outputs_dir / value for key, value in params["dirs"].items()},
    )

    key_map = NestedKeyCodec.from_key_map(params["key_map"])

    metric = varifold_metric_from_results(results, backend=backend)

    # get relevant dirs
    dataset = collect_dataset(dir_config.meshes_dir, key_map.keys(encoded=True))
    local_regs = collect_local_registrations(
        dir_config.registration_dir, key_map.keys(encoded=True)
    )
    global_shoots = collect_global_shoots(
        dir_config.shoot_dir, key_map.keys(encoded=True)
    )
    transports = collect_transports(
        dir_config.transport_dir, key_map.keys(encoded=True)
    )

    atlases = collect_atlases(dir_config.atlas_dir, key_map.keys(encoded=True))
    global_atlas = get_global_atlas(dir_config.atlas_dir)

    # compute errors
    if include_local_rec_error:
        local_errors = local_regs.map_values(
            reconstruction_error,
            dist_fnc=metric.dist,
        )
        save_indexed_array(dists_folder / "rec_local.npz", local_errors.flatten())

    if include_atlas_rec_error:
        atlases_errors = atlases.map_values(
            atlas_reconstruction_error,
            dist_fnc=metric.dist,
        )
        save_indexed_array(
            dists_folder / "rec_atlas.npz", NestedDataset(atlases_errors.data).flatten()
        )

    if include_global_atlas_rec_error:
        atlas_errors = atlas_reconstruction_error(
            global_atlas,
            dist_fnc=metric.dist,
        )
        save_indexed_array(
            dists_folder / "rec_global_atlas.npz",
            atlas_errors,
        )

    if include_transport_error:
        transport_error = transports.map_values(
            parallel_transport_dir_error,
            atlas=global_atlas.template().as_pv_surface(),
            dist_fnc=metric.dist,
        )
        save_indexed_array(
            dists_folder / "rec_transport.npz", transport_error.flatten()
        )

    # pairwise distances
    if include_local_pairwise:
        dists = pairwise_dist(dataset, metric.dist)
        dists.save(dists_folder / "local_pairwise")

    if include_rec_local_pairwise:
        dists = local_pairwise_dist(local_regs, metric.dist)
        dists.save(dists_folder / "rec_local_pairwise")

    if include_global_pairwise:
        dists = global_pairwise_dist(global_shoots, metric.dist)
        dists.save(dists_folder / "global_pairwise")

from functools import cached_property
from pathlib import Path

import geomstats.backend as gs

from polpo.dataset import Dataset, NestedDataset
from polpo.distmat import PairwiseDistances
from polpo.io.json import load_json
from polpo.numpy.io import save_dict_as_array
from polpo.surface_mesh.deformetrica.core import (
    DeterministicAtlasDir,
    Point,
    RegistrationDir,
    ShootDir,
    TransportDir,
)
from polpo.surface_mesh.deformetrica.utils import DirConfig
from polpo.surface_mesh.varifold.geometry import VarifoldMetric
from polpo.utils import NestedKeyCodec
from polpo.utils.np import pairwise_dists
from polpo.workflow.task import TaskRunner


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


class PostDistances(TaskRunner):
    def __init__(
        self,
        experiment_dir,
        results_dir="post_dists",
        backend="auto",
    ):
        self.experiment_dir = Path(experiment_dir)

        results_dir = Path(results_dir)
        if not results_dir.is_absolute():
            results_dir = self.experiment_dir / results_dir

        self.backend = backend
        super().__init__(
            results_dir,
            metadata={
                "backend": backend,
                "experiment_dir": str(self.experiment_dir),
            },
        )

    def tasks(self):
        return {
            "local_rec_error": self.local_rec_error,
            "atlas_rec_error": self.atlas_rec_error,
            "global_atlas_rec_error": self.global_atlas_rec_error,
            "transport_error": self.transport_error,
            "local_pairwise": self.local_pairwise,
            "rec_local_pairwise": self.rec_local_pairwise,
            "global_pairwise": self.global_pairwise,
        }

    @cached_property
    def params(self):
        return load_json(self.experiment_dir / "params.json")

    @cached_property
    def results(self):
        return load_json(self.experiment_dir / "results.json")

    @cached_property
    def dir_config(self):
        return DirConfig(
            outputs_dir=self.experiment_dir,
            **{
                key: self.experiment_dir / value
                for key, value in self.params["dirs"].items()
            },
        )

    @cached_property
    def key_map(self):
        return NestedKeyCodec.from_key_map(self.params["metadata"]["key_map"])

    @cached_property
    def encoded_keys(self):
        return self.key_map.keys(encoded=True)

    @cached_property
    def metric(self):
        metric = varifold_metric_from_results(
            self.results,
            backend=self.backend,
        )

        self.set_resolved(
            geomstats_backend=gs.__name__,
            device="gpu" if metric._gpu else "cpu",
        )

        return metric

    @cached_property
    def dataset(self):
        return collect_dataset(
            self.dir_config.meshes_dir,
            self.encoded_keys,
        )

    @cached_property
    def local_regs(self):
        return collect_local_registrations(
            self.dir_config.registration_dir,
            self.encoded_keys,
        )

    @cached_property
    def local_rec_points(self):
        # TODO: use to compute distances
        return self.local_regs.map_values(lambda x: x.reconstructed())

    @cached_property
    def global_shoots(self):
        return collect_global_shoots(
            self.dir_config.shoot_dir,
            self.encoded_keys,
        )

    @cached_property
    def global_points(self):
        # TODO: use to compute distances
        return self.global_shoots.map_values(
            lambda x: x.point(),
        )

    @cached_property
    def transports(self):
        return collect_transports(
            self.dir_config.transport_dir,
            self.encoded_keys,
        )

    @cached_property
    def atlases(self):
        return collect_atlases(
            self.dir_config.atlas_dir,
            self.encoded_keys,
        )

    @cached_property
    def global_atlas(self):
        return get_global_atlas(self.dir_config.atlas_dir)

    def local_rec_error(self):
        errors = self.local_regs.map_values(
            reconstruction_error,
            dist_fnc=self.metric.dist,
        )
        save_dict_as_array(
            self.results_dir / "rec_local",
            errors.flatten(),
        )

    def atlas_rec_error(self):
        errors = self.atlases.map_values(
            atlas_reconstruction_error,
            dist_fnc=self.metric.dist,
        )
        save_dict_as_array(
            self.results_dir / "rec_atlas",
            NestedDataset(errors.data).flatten(),
        )

    def global_atlas_rec_error(self):
        errors = atlas_reconstruction_error(
            self.global_atlas,
            dist_fnc=self.metric.dist,
        )
        save_dict_as_array(
            self.results_dir / "rec_global_atlas",
            errors,
        )

    def transport_error(self):
        errors = self.transports.map_values(
            parallel_transport_dir_error,
            atlas=self.global_atlas.template().as_pv_surface(),
            dist_fnc=self.metric.dist,
        )
        save_dict_as_array(
            self.results_dir / "rec_transport",
            errors.flatten(),
        )

    def local_pairwise(self):
        distances = pairwise_dist(
            self.dataset,
            self.metric.dist,
        )
        distances.save(self.results_dir / "local_pairwise")

    def rec_local_pairwise(self):
        distances = local_pairwise_dist(
            self.local_regs,
            self.metric.dist,
        )
        distances.save(self.results_dir / "rec_local_pairwise")

    def global_pairwise(self):
        distances = global_pairwise_dist(
            self.global_shoots,
            self.metric.dist,
        )
        distances.save(self.results_dir / "global_pairwise")

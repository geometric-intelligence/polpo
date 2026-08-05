from collections.abc import Mapping
from functools import cached_property
from pathlib import Path

import geomstats.backend as gs

from polpo.dataset import Dataset
from polpo.distmat import PairwiseDistances
from polpo.numpy.io import load_dict, save_dict_as_array
from polpo.surface_mesh.euclidean import EuclideanSurfaces
from polpo.surface_mesh.varifold.geometry import VarifoldMetric
from polpo.utils.np import pairwise_dists
from polpo.workflow.task import TaskRunner

from .run import LddmmToGlobalRun

# TODO: add ability to confirm run success

# TODO: centralize tasks


def varifold_metric_from_results(data, backend="auto"):
    sigma = data["kernel_tuning"]["sigma_var"]
    return VarifoldMetric(sigma=sigma, backend=backend)


def _reconstruction_error(registration_res, dist_fnc):
    return dist_fnc(
        registration_res.point.as_pv_surface(),
        registration_res.reconstructed().as_pv_surface(),
    )


def _atlas_reconstruction_error(atlas_res, dist_fnc):
    def _id_to_key(id_):
        if "-" in id_:
            return tuple(id_.split("-"))

        return id_

    return {
        _id_to_key(point.id): dist_fnc(point.as_pv_surface(), cmp_point.as_pv_surface())
        for point, cmp_point in zip(atlas_res.points, atlas_res.reconstructed())
    }


def _parallel_transport_res_error(transport_res, atlas, dist_fnc):
    return dist_fnc(
        transport_res.reconstructed().as_pv_surface(),
        atlas,
    )


class LddmmToGlobalDistanceEvaluator:
    # original means after rigid alignment
    def __init__(self, source, metric):
        self.source = source
        self.metric = metric

    def local_reconstruction_error(self):
        # compares original against reconstructed after registration
        return self.source.encoded.local_registrations.map_values(
            _reconstruction_error,
            dist_fnc=self.metric.dist,
        ).flatten()

    def local_atlas_reconstruction_error(self):
        # compares original against reconstructed during deterministic atlas
        errors = self.source.encoded.local_atlases.map_values(
            _atlas_reconstruction_error,
            dist_fnc=self.metric.dist,
        )
        return Dataset({k: v for d in errors.values_list() for k, v in d.items()})

    def global_atlas_reconstruction_error(self):
        return _atlas_reconstruction_error(
            self.source.global_atlas,
            dist_fnc=self.metric.dist,
        )

    def transport_error(self):
        return self.source.encoded.transports.map_values(
            _parallel_transport_res_error,
            atlas=self.source.global_atlas_point.as_pv_surface(),
            dist_fnc=self.metric.dist,
        ).flatten()

    def local_pairwise(self):
        return self._pairwise(self.source.encoded.dataset.flatten())

    def local_reconstruction_pairwise(self):
        return self._pairwise(self.source.encoded.local_reconstructed_points.flatten())

    def global_pairwise(self):
        return self._pairwise(self.source.encoded.global_points.flatten())

    def _pairwise(self, data):
        surfaces = data.map_values(lambda point: point.as_pv_surface())

        return PairwiseDistances(
            surfaces.keys_list(),
            pairwise_dists(
                surfaces.values_list(),
                self.metric.dist,
                as_matrix=False,
            ),
        )


class LddmmToGlobalDistances(TaskRunner):
    def __init__(
        self,
        experiment_dir,
        results_dir="post_dists",
        backend="auto",
    ):
        self.source = LddmmToGlobalRun(experiment_dir)

        results_dir = Path(results_dir)
        if not results_dir.is_absolute():
            results_dir = experiment_dir / results_dir

        self.backend = backend
        super().__init__(
            results_dir,
            metadata={
                "backend": backend,
                "experiment_dir": str(experiment_dir),
            },
        )

    @cached_property
    def evaluator(self):
        return LddmmToGlobalDistanceEvaluator(
            self.source,
            self.metric,
        )

    def tasks(self):
        return {
            "local_reconstruction_error": self.local_reconstruction_error,
            "local_atlas_reconstruction_error": (self.local_atlas_reconstruction_error),
            "global_atlas_reconstruction_error": (
                self.global_atlas_reconstruction_error
            ),
            "transport_error": self.transport_error,
            "local_pairwise": self.local_pairwise,
            "reconstructed_local_pairwise": (self.reconstructed_local_pairwise),
            "global_pairwise": self.global_pairwise,
        }

    @cached_property
    def metric(self):
        metric = varifold_metric_from_results(
            self.source.results,
            backend=self.backend,
        )

        self.set_resolved(
            geomstats_backend=gs.__name__,
            device="gpu" if metric._gpu else "cpu",
        )

        return metric

    def local_reconstruction_error(self):
        errors = self.evaluator.local_reconstruction_error()
        save_dict_as_array(
            self.results_dir / "rec_local",
            errors,
        )

    def local_atlas_reconstruction_error(self):
        errors = self.evaluator.local_atlas_reconstruction_error()
        save_dict_as_array(
            self.results_dir / "rec_local_atlas",
            errors.flatten(),
        )

    def global_atlas_reconstruction_error(self):
        errors = self.evaluator.global_atlas_reconstruction_error()
        save_dict_as_array(
            self.results_dir / "rec_global_atlas",
            errors,
        )

    def transport_error(self):
        errors = self.evaluator.transport_error()
        save_dict_as_array(
            self.results_dir / "rec_transport",
            errors,
        )

    def local_pairwise(self):
        distances = self.evaluator.local_pairwise()
        distances.save(self.results_dir / "local_pairwise")

    def reconstructed_local_pairwise(self):
        distances = self.evaluator.reconstructed_local_pairwise()
        distances.save(self.results_dir / "rec_local_pairwise")

    def global_pairwise(self):
        distances = self.evaluator.global_pairwise()
        distances.save(self.results_dir / "global_pairwise")

    @property
    def results(self):
        return StoredDistanceResults(self.results_dir)


class EuclideanLddmmToGlobalDistances:
    def __init__(self, experiment_dir):
        self.source = LddmmToGlobalRun(experiment_dir)

        self.results_ = None

    def tasks(self):
        evaluator = self.evaluator

        return {
            "local_reconstruction_error": (evaluator.local_reconstruction_error),
            "local_atlas_reconstruction_error": (
                evaluator.local_atlas_reconstruction_error
            ),
            "global_atlas_reconstruction_error": (
                evaluator.global_atlas_reconstruction_error
            ),
            "transport_error": evaluator.transport_error,
            "local_pairwise": evaluator.local_pairwise,
            "local_reconstruction_pairwise": evaluator.local_reconstruction_pairwise,
            "global_pairwise": evaluator.global_pairwise,
        }

    @cached_property
    def evaluator(self):
        return LddmmToGlobalDistanceEvaluator(
            self.source,
            metric=EuclideanSurfaces(
                faces=self.source.global_atlas_point.as_pv_surface().faces
            ).metric,
        )

    def run(self, tasks=None):
        available = self.tasks()

        if tasks is None:
            tasks = list(available)

        unknown = set(tasks) - set(available)
        if unknown:
            raise ValueError(f"Unknown tasks: {sorted(unknown)}")

        self.results_ = InMemoryDistanceResults(
            {task: available[task]() for task in tasks}
        )

        return self

    @property
    def results(self):
        if self.results_ is None:
            raise RuntimeError("Distances have not been computed. Call run() first.")

        return self.results_


class DistanceResults(Mapping):
    # TODO: check need for this
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class InMemoryDistanceResults(DistanceResults):
    def __init__(self, data):
        self._data = dict(data)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


class StoredDistanceResults(DistanceResults):
    LOADERS = {
        "local_reconstruction_error": (
            "rec_local",
            load_dict,
        ),
        "local_atlas_reconstruction_error": (
            "rec_local_atlas",
            load_dict,
        ),
        "global_atlas_reconstruction_error": (
            "rec_global_atlas",
            load_dict,
        ),
        "transport_error": (
            "rec_transport",
            load_dict,
        ),
        "local_pairwise": (
            "local_pairwise",
            PairwiseDistances.load,
        ),
        "local_reconstruction_pairwise": (
            "rec_local_pairwise",
            PairwiseDistances.load,
        ),
        "global_pairwise": (
            "global_pairwise",
            PairwiseDistances.load,
        ),
    }

    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self._cache = {}

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(key)

        if key not in self._cache:
            filename, loader = self.LOADERS[key]
            self._cache[key] = loader(self.results_dir / filename)

        return self._cache[key]

    def __iter__(self):
        return (key for key in self.LOADERS if self._exists(key))

    def __len__(self):
        return sum(1 for _ in self)

    def __contains__(self, key):
        return key in self.LOADERS and self._exists(key)

    def _exists(self, key):
        filename, _ = self.LOADERS[key]
        return self._result_exists(self.results_dir / filename)

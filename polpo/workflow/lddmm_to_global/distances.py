from collections.abc import Mapping
from functools import cached_property, partial
from pathlib import Path

import geomstats.backend as gs

from polpo.dataset import Dataset
from polpo.distmat import PairwiseDistances
from polpo.io.json import load_json
from polpo.numpy.io import load_dict, save_dict_as_array
from polpo.surface_mesh.euclidean import EuclideanSurfaces
from polpo.surface_mesh.varifold.geometry import VarifoldMetric
from polpo.utils.dict_ import merge_dicts
from polpo.utils.np import pairwise_dists
from polpo.workflow.task import TaskRunner

from .output import LddmmToGlobalOutput

try:
    # TODO: fix this at lddmmmetric level?
    from polpo.surface_mesh.deformetrica.geometry import LddmmMetric
except ImportError:
    pass


def _save_pairwise_distances(path, result):
    result.save(path)


def _load_distances(path):
    return Dataset(load_dict(path))


def _task(key, filename=None, *, pairwise=False):
    filename = f"{key}.npz" if filename is None else filename

    save = _save_pairwise_distances if pairwise else save_dict_as_array
    load = PairwiseDistances.load if pairwise else _load_distances

    return key, {
        "filename": filename,
        "save": save,
        "load": load,
    }


REGISTRATION_TASKS = dict(
    [
        _task("local_atlas_to_reconstructed"),
        _task("global_atlas_to_global"),
    ]
)

DISTANCE_TASKS = dict(
    [
        _task("local_reconstruction_error"),
        _task("local_atlas_fit_error"),
        _task("global_atlas_fit_error"),
        _task("local_to_global_reconstruction_error"),
        _task("local_to_global_transport_error"),
        _task("local_pairwise", pairwise=True),
        _task("local_reconstructed_pairwise", pairwise=True),
        _task("global_pairwise", pairwise=True),
    ]
)


def varifold_metric_from_results(data, backend="auto"):
    sigma = data["kernel_tuning"]["sigma_var"]
    return VarifoldMetric(sigma=sigma, backend=backend)


def _reconstruction_error(registration_res, dist_fnc):
    return dist_fnc(
        registration_res.point.as_pv_surface(),
        registration_res.reconstructed().as_pv_surface(),
    )


def _reconstruction_error_lddmm(registration_res, dist_fnc):
    return dist_fnc(
        registration_res.tangent_vec(),
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


class Evaluator:
    task_specs = ()

    def __init__(self, source, metric):
        self.source = source
        self.metric = metric

    def tasks(self):
        return {name: getattr(self, name) for name in self.task_specs}

    def run(self, tasks=None):
        available = self.tasks()

        if tasks is None:
            tasks = list(available)

        unknown = set(tasks) - set(available)
        if unknown:
            raise ValueError(f"Unknown tasks: {sorted(unknown)}")

        self.results_ = InMemoryResults({task: available[task]() for task in tasks})

        return self

    @property
    def results(self):
        if self.results_ is None:
            raise RuntimeError("Distances have not been computed. Call run() first.")

        return self.results_

    @property
    def requested(self):
        return {}

    @property
    def resolved(self):
        return {}


class DistanceEvaluator(Evaluator):
    # original means after rigid alignment

    task_specs = DISTANCE_TASKS

    def local_reconstruction_error(self):
        # compares original against reconstructed after registration
        return self.source.encoded.local_registrations.map_values(
            _reconstruction_error,
            dist_fnc=self.metric.dist,
        ).flatten()

    def local_atlas_fit_error(self):
        # compares original against reconstructed during deterministic atlas
        errors = self.source.encoded.local_atlases.map_values(
            _atlas_reconstruction_error,
            dist_fnc=self.metric.dist,
        )
        return Dataset(merge_dicts(errors.values_list()))

    def global_atlas_fit_error(self):
        # compares local atlas against reconstructed local atlas during deterministic atlas
        return _atlas_reconstruction_error(
            self.source.global_atlas,
            dist_fnc=self.metric.dist,
        )

    def local_to_global_reconstruction_error(self):
        # compares global against registration from local
        # establishes transport direction
        return self.source.encoded.registrations_to_global_atlas.map_values(
            _reconstruction_error,
            dist_fnc=self.metric.dist,
        )

    def local_to_global_transport_error(self):
        # error induced by transport direction
        # only collecting one per outer due to the nature of the algorithm
        # must compare with local_to_global_reconstruction_error

        # TODO: improve
        # NB: only fan as reconstructed
        for _ in range(100):
            trans_res = (
                self.source.encoded.transports.sample_inner(n_samples=1)
                .flatten()
                .map_keys(lambda x: x[0])
            )
            if trans_res.apply(
                lambda values: all([res.method == "fan" for res in values])
            ):
                break
        else:
            raise ValueError("Oops, are you sure transport uses fan?")

        return trans_res.map_values(
            _parallel_transport_res_error,
            atlas=self.source.global_atlas_point.as_pv_surface(),
            dist_fnc=self.metric.dist,
        )

    def local_pairwise(self):
        return self._pairwise(self.source.encoded.dataset.flatten())

    def local_reconstructed_pairwise(self):
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


class VarifoldDistances(DistanceEvaluator):
    def __init__(self, experiment_dir, backend="auto"):
        source = LddmmToGlobalOutput(experiment_dir)

        metric = varifold_metric_from_results(
            source.results,
            backend=backend,
        )
        self.backend = backend

        super().__init__(source, metric)

    @property
    def requested(self):
        return {"backend": self.backend}

    @property
    def resolved(self):
        return dict(
            geomstats_backend=(gs.__name__,),
            device=("gpu" if self.metric._gpu else "cpu",),
        )


class EuclideanDistances(DistanceEvaluator):
    def __init__(self, experiment_dir):
        source = LddmmToGlobalOutput(experiment_dir)

        metric = EuclideanSurfaces(
            faces=source.global_atlas_point.as_pv_surface().faces
        ).metric

        super().__init__(source, metric)


class LddmmDistances(Evaluator):
    task_specs = REGISTRATION_TASKS

    def __init__(self, experiment_dir):
        source = LddmmToGlobalOutput(experiment_dir)
        metric = LddmmMetric(
            experiment_dir,
            kernel_width=source.results["kernel_tuning"]["sigma_vel"],
        )

        super().__init__(source, metric)

    def local_atlas_to_reconstructed(self):
        # distance from local atlas to reconstructed
        return self.source.encoded.local_registrations.map_values(
            # TODO: add norm
            lambda x: self.metric.norm(x.tangent_vec()),
        ).flatten()

    def global_atlas_to_global(self):
        # distance from local atlas to reconstructed
        # NB: parallel transport preserves distance
        return self.source.encoded.global_shoots.map_values(
            lambda x: self.metric.norm(x.tangent_vec),
        ).flatten()


class PersistentEvaluator(TaskRunner):
    def __init__(self, evaluator, results_dir="post_dists"):
        results_dir = Path(results_dir)
        if not results_dir.is_absolute():
            results_dir = evaluator.source.path / results_dir

        super().__init__(results_dir, metadata=evaluator.requested)

        self.evaluator = evaluator
        self.set_resolved(**evaluator.resolved)

    @property
    def results(self):
        return DistanceResults(self.results_dir, task_specs=self.evaluator.task_specs)

    def tasks(self):
        return {
            name: partial(self._compute_and_save, name)
            for name in self.evaluator.tasks()
        }

    def _compute_and_save(self, task):
        result = self.evaluator.tasks()[task]()

        spec = self.evaluator.task_specs[task]
        path = self.results_dir / spec["filename"]
        spec["save"](path, result)


class _Results(Mapping):
    def __init__(self, label_map=None):
        self.label_map = None

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def _transform(self, data):
        if self.label_map is None:
            return data

        if isinstance(data, PairwiseDistances):
            return data.map_labels(self.label_map)

        if isinstance(data, Dataset):
            return data.map_keys(self.label_map)

        return data


class InMemoryResults(_Results):
    def __init__(self, data, label_map=None):
        super().__init__(label_map=label_map)
        self._data = dict(data)

    def __getitem__(self, key):
        return self._transform(self._data[key])

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def with_label_map(self, label_map):
        return self.__class__(
            self.data,
            label_map=label_map,
        )


class DistanceResults(_Results):
    def __init__(self, results_dir, task_specs=None, label_map=None):
        self.results_dir = Path(results_dir)
        self.label_map = label_map

        if task_specs is None:
            task_specs = (
                DISTANCE_TASKS
                if "global_pairwise" in self.manifest["tasks"]
                else REGISTRATION_TASKS
            )

        self.task_specs = task_specs

        self._cache = {}

    def with_label_map(self, label_map):
        return self.__class__(
            self.results_dir,
            task_specs=self.task_specs,
            label_map=label_map,
        )

    @property
    def manifest_path(self):
        return self.results_dir / "manifest.json"

    @cached_property
    def manifest(self):
        return load_json(self.manifest_path)

    def __getitem__(self, task):
        if task not in self.task_specs:
            raise KeyError(task)

        if not self.is_available(task):
            raise KeyError(f"Distance result {task!r} is not available.")

        if task not in self._cache:
            spec = self.task_specs[task]
            path = self.results_dir / spec["filename"]
            self._cache[task] = self._transform(spec["load"](path))

        return self._cache[task]

    def __iter__(self):
        return (task for task in self.task_specs if self.is_available(task))

    def __len__(self):
        return sum(self.is_available(task) for task in self.task_specs)

    def __contains__(self, task):
        return self.is_available(task)

    def is_available(self, task):
        if task not in self.task_specs:
            return False

        task_info = self.manifest.get("tasks", {}).get(task, {})
        return task_info.get("status") == "completed"

    def clear_cache(self, task=None):
        if task is None:
            self._cache.clear()
            return

        self._cache.pop(task, None)

    def refresh(self):
        self.clear_cache()
        self.__dict__.pop("manifest", None)

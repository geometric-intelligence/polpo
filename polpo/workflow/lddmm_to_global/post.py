from functools import cached_property
from pathlib import Path

import geomstats.backend as gs

from polpo.distmat import PairwiseDistances
from polpo.numpy.io import save_dict_as_array
from polpo.surface_mesh.varifold.geometry import VarifoldMetric
from polpo.utils.np import pairwise_dists
from polpo.workflow.task import TaskRunner

from .run import LddmmToGlobalRun

# TODO: add ability to confirm run success


def varifold_metric_from_results(data, backend="auto"):
    sigma = data["kernel_tuning"]["sigma_var"]
    return VarifoldMetric(sigma=sigma, backend=backend)


def _reconstruction_error(registration_res, dist_fnc):
    return dist_fnc(
        registration_res.point.as_pv_surface(),
        registration_res.reconstructed().as_pv_surface(),
    )


def _atlas_reconstruction_error(atlas_res, dist_fnc):
    return {
        point.id: dist_fnc(point.as_pv_surface(), cmp_point.as_pv_surface())
        for point, cmp_point in zip(atlas_res.points, atlas_res.reconstructed())
    }


def _parallel_transport_res_error(transport_res, atlas, dist_fnc):
    return dist_fnc(
        transport_res.reconstructed().as_pv_surface(),
        atlas,
    )


class LddmmToGlobalDistances(TaskRunner):
    def __init__(
        self,
        experiment_dir,
        results_dir="post_dists",
        backend="auto",
    ):
        self.run = LddmmToGlobalRun(experiment_dir)

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
            self.run.results,
            backend=self.backend,
        )

        self.set_resolved(
            geomstats_backend=gs.__name__,
            device="gpu" if metric._gpu else "cpu",
        )

        return metric

    def local_reconstruction_error(self):
        errors = self.run.encoded.local_registrations.map_values(
            _reconstruction_error,
            dist_fnc=self.metric.dist,
        )
        save_dict_as_array(
            self.results_dir / "rec_local",
            errors.flatten(),
        )
        return errors

    def local_atlas_reconstruction_error(self):
        errors = self.run.encoded.local_atlases.map_values(
            _atlas_reconstruction_error,
            dist_fnc=self.metric.dist,
        )
        save_dict_as_array(
            self.results_dir / "rec_local_atlas",
            errors.flatten(),
        )

    def global_atlas_reconstruction_error(self):
        errors = _atlas_reconstruction_error(
            self.run.global_atlas,
            dist_fnc=self.metric.dist,
        )
        save_dict_as_array(
            self.results_dir / "rec_global_atlas",
            errors,
        )

    def transport_error(self):
        errors = self.run.encoded.transports.map_values(
            _parallel_transport_res_error,
            atlas=self.run.global_atlas_point.as_pv_surface(),
            dist_fnc=self.metric.dist,
        )
        save_dict_as_array(
            self.results_dir / "rec_transport",
            errors.flatten(),
        )

    def local_pairwise(self):
        return self._compute_pairwise_dists(
            self.run.encoded.dataset.flatten().map_values(lambda x: x.as_pv_surface()),
            filename="local_pairwise",
        )

    def reconstructed_local_pairwise(self):
        return self._compute_pairwise_dists(
            self.run.encoded.local_reconstructed_points.flatten().map_values(
                lambda x: x.as_pv_surface()
            ),
            filename="rec_local_pairwise",
        )

    def global_pairwise(self):
        return self._compute_pairwise_dists(
            self.run.encoded.global_points.flatten().map_values(
                lambda x: x.as_pv_surface()
            ),
            filename="global_pairwise",
        )

    def _compute_pairwise_dists(self, dataset, filename=None):
        distances = PairwiseDistances(
            dataset.keys_list(),
            pairwise_dists(dataset.values_list(), self.metric.dist, as_matrix=False),
        )

        if filename is not None:
            distances.save(self.results_dir / filename)

        return distances

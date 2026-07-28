import json

import geomstats.backend as gs

from polpo.dataset import NestedDataset
from polpo.protocol.mixin import PairwiseDistancesMixin, RigidAlignmentMixin
from polpo.surface_mesh.varifold.tuning.metric_based import SigmaFromLengths
from polpo.time import Timer


class PairwiseVarifold(RigidAlignmentMixin, PairwiseDistancesMixin):
    def __init__(
        self,
        known_correspondences,
        results_dir,
        ratio_charlen_mesh=2.0,
        ratio_charlen=0.25,
        n_jobs=1,
        backend="keops",
    ):
        # TODO: add longitudinal param? affects tune
        # TODO: worth creating
        self.timer = Timer()

        self.known_correspondences = known_correspondences
        self.results_dir = results_dir

        self.ratio_charlen = ratio_charlen
        self.ratio_charlen_mesh = ratio_charlen_mesh

        self.n_jobs = n_jobs
        self.backend = backend

        self.reset()

    def reset(self):
        self.results_ = {"version": "0.1.0"}
        self.params_ = {}
        self.timer.reset()

        self.dists_ = None

    def tune_kernel(self, nested_meshes):
        # select varifold kernel using a randomly selected mesh per subject
        self.timer.start("tuning")

        sigma_search = SigmaFromLengths(
            ratio_charlen_mesh=self.ratio_charlen_mesh,
            ratio_charlen=self.ratio_charlen,
            backend=self.backend,
        )

        # TODO: update if longitudinal
        # TODO: add random_state?
        selected_meshes = nested_meshes.sample_inner()
        sigma_search.fit(selected_meshes.flatten().values_list())

        self.timer.stop("tuning")
        metric = sigma_search.optimal_metric_

        self.params_["kernel_tuning"] = {
            "ratio_charlen_mesh": self.ratio_charlen_mesh,
            "ratio_charlen": self.ratio_charlen,
        }

        self.params_["metric"] = {
            "metric": "varifold",
            "geomstats_backend": gs.__name__,
            "backend": self.backend,
        }

        self.results_["kernel_tuning"] = {
            "sigma": float(sigma_search.sigma_),  # for serialization
            "meshes": selected_meshes.flatten().keys_list(),
        }
        self.results_["metric"] = {
            "device": "gpu" if metric._gpu else "cpu",
        }
        return metric

    def write(self):
        with open(self.results_dir / "params.json", "w") as file:
            json.dump(self.params_, file, indent=2)

        with open(self.results_dir / "results.json", "w") as file:
            json.dump(self.results_, file, indent=2)

        self.dists_.save(self.results_dir / "pairwise_dists")

        with open(self.results_dir / "time.json", "w") as file:
            json.dump(self.timer.as_dict(), file, indent=2)

    def run(self, nested_meshes):
        # nested_meshes: dict or polpo.dataset.NestedDataset
        if isinstance(nested_meshes, dict):
            nested_meshes = NestedDataset(nested_meshes)

        self.reset()

        self.timer.start("run")
        nested_meshes = self.preprocess_meshes(nested_meshes.flatten()).nest()

        metric = self.tune_kernel(nested_meshes)
        self.dists_ = self.compute_pairwise_dists(nested_meshes.flatten(), metric)

        self.timer.stop("run")

        self.write()

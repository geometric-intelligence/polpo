import json

import geomstats.backend as gs
import numpy as np

import polpo.utils as putils
from polpo.protocol.mixin import MeshPreprocessorMixin
from polpo.surface_mesh.varifold.tuning.metric_based import SigmaFromLengths
from polpo.time import Timer


class PairwiseVarifold(MeshPreprocessorMixin):
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
        mesh_per_outer = []
        keys = []
        for outer_key, meshes in nested_meshes.items():
            inner_key = putils.extract_random_key(meshes)
            mesh_per_outer.append(meshes[inner_key])

            keys.append(f"{outer_key}-{inner_key}")

        sigma_search.fit(mesh_per_outer)

        metric = sigma_search.optimal_metric_
        self.timer.stop("tuning")

        self.results_["kernel_tuning"] = {
            "sigma": float(sigma_search.sigma_),  # for serialization
            "meshes": keys,
        }

        self.params_["kernel_tuning"] = {
            "ratio_charlen_mesh": self.ratio_charlen_mesh,
            "ratio_charlen": self.ratio_charlen,
        }

        return metric

    def compute_dists(self, meshes, metric):
        # flatten
        self.timer.start("dists")

        meshes_flat = putils.unnest_dict(meshes, sep="-")
        self.results_["keys"] = list(meshes_flat.keys())

        dists = putils.pairwise_dists_par(
            list(meshes_flat.values()),
            metric.dist,
            n_jobs=self.n_jobs,
        )

        self.timer.stop("dists")

        self.params_["dists"] = {
            "n_jobs": self.n_jobs,
            "geomstats_backend": gs.__name__,
            "backend": self.backend,
        }

        self.results_["dists"] = {
            "device": "gpu" if metric._gpu else "cpu",
        }

        return dists

    def write(self):
        with open(self.results_dir / "params.json", "w") as file:
            json.dump(self.params_, file, indent=2)

        with open(self.results_dir / "results.json", "w") as file:
            json.dump(self.results_, file, indent=2)

        # TODO: can dump only upper triangular?
        # TODO: PairwiseDistances save
        self.dists_.save(self.results_dir / "pairwise_dists")

        with open(self.results_dir / "time.json", "w") as file:
            json.dump(self.timer.as_dict(), file, indent=2)

    def run(self, nested_meshes):
        # nested_meshes: NestedDataset
        self.reset()

        self.timer.start("run")

        nested_meshes = self.preprocess_meshes(nested_meshes)
        metric = self.tune_kernel(nested_meshes)
        self.dists_ = self.compute_dists(nested_meshes, metric)

        self.timer.stop("run")

        self.write()

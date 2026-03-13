import json
import time

import numpy as np

import polpo.preprocessing.dict as ppdict
import polpo.utils as putils
from polpo.mesh.surface import PvSurface
from polpo.mesh.varifold.tuning import SigmaFromLengths
from polpo.preprocessing.mesh.registration import RigidAlignment


class Timer:
    def __init__(self):
        self.times = {}

    def stamp(self, key):
        self.times[key] = time.perf_counter()

    def elapsed(self, start, end):
        return self.times[end] - self.times[start]

    def reset(self):
        self.times = {}

    def as_dict(self):
        return self.times


class PairwiseVarifold:
    def __init__(self, mesh_loader, known_correspondences, results_dir):
        if not callable(mesh_loader):
            mesh_loader = lambda *args: mesh_loader

        self.timer = Timer()

        self.mesh_loader = mesh_loader
        self.known_correspondences = known_correspondences
        self.results_dir = results_dir

        self.reset()

    def reset(self):
        self.results_ = {"version": "0.1.0"}
        self.dists_ = None
        self.timer.reset()

    def load_data(self):
        self.timer.stamp("load_start")
        raw_meshes = self.mesh_loader()  # subject, session
        self.timer.stamp("load_end")

        return raw_meshes

    def preprocess_meshes(self, raw_meshes):
        self.timer.stamp("prep_start")

        align_pipe = RigidAlignment(
            target=ppdict.ExtractRandomKey()(putils.get_first(raw_meshes)),
            known_correspondences=self.known_correspondences,
        )

        meshes = (ppdict.DictMap(align_pipe + ppdict.DictMap(PvSurface)))(raw_meshes)

        self.timer.stamp("prep_end")

        return meshes

    def tune_kernel(self, meshes):
        # select varifold kernel
        self.timer.stamp("tuning_start")

        sigma_search = SigmaFromLengths(
            ratio_charlen_mesh=2.0,
            ratio_charlen=0.25,
        )

        mesh_per_subject = (
            ppdict.DictMap(ppdict.ExtractRandomKey()) + ppdict.DictToValuesList()
        )(meshes)  # one mesh per subject

        sigma_search.fit(mesh_per_subject)
        self.results_["sigma"] = sigma_search.sigma_

        metric = sigma_search.optimal_metric_
        self.timer.stamp("tuning_end")

        return metric

    def compute_dist(self, meshes, metric):
        # flatten
        self.timer.stamp("dist_start")

        meshes_flat = ppdict.UnnestDict(sep="-")(meshes)
        self.results_["keys"] = list(meshes_flat.keys())

        dists = putils.pairwise_dists(list(meshes_flat.values()), metric)

        self.timer.stamp("dist_end")

        return dists

    def write(self):
        self.results_["time"] = self.timer.as_dict()

        with open(self.results_dir / "results.json", "w") as file:
            json.dump(self.results_, file, indent=4)

        # TODO: can dump only upper triangular?
        np.save(self.results_dir / "pair_dists.npy", self.dists_)

    def run(self):
        self.reset()

        self.timer.stamp("start")

        raw_meshes = self.load_data()
        meshes = self.preprocess_meshes(raw_meshes)
        metric = self.tune_kernel(meshes)
        self.dists_ = self.compute_dist(meshes, metric)

        self.write()

        self.timer.stamp("end")

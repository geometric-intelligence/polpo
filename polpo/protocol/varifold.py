import json

import numpy as np

import polpo.preprocessing.dict as ppdict
import polpo.utils as putils
from polpo.mesh.surface import PvSurface
from polpo.mesh.varifold.tuning import SigmaFromLengths
from polpo.preprocessing.mesh.registration import RigidAlignment
from polpo.time import Timer

# TODO: add documentation


class PairwiseVarifold:
    def __init__(self, mesh_loader, known_correspondences, results_dir, timer=None):
        if not callable(mesh_loader):
            mesh_loader = lambda *args: mesh_loader

        if timer is None:
            timer = Timer()

        self.timer = timer

        self.mesh_loader = mesh_loader
        self.known_correspondences = known_correspondences
        self.results_dir = results_dir

        self.reset()

    def reset(self):
        self.results_ = {"version": "0.1.0"}
        self.dists_ = None
        self.timer.reset()

    def load_data(self):
        with self.timer("load"):
            raw_meshes = self.mesh_loader()  # subject, session

        return raw_meshes

    def preprocess_meshes(self, raw_meshes):
        self.timer.start("prep")

        # TODO: improve naming
        # TODO: save info
        random_subj_key = None
        align_pipe = RigidAlignment(
            target=ppdict.ExtractRandomKey()(putils.get_first(raw_meshes)),
            known_correspondences=self.known_correspondences,
        )

        meshes = (ppdict.DictMap(align_pipe + ppdict.DictMap(PvSurface)))(raw_meshes)

        self.timer.stop("prep")

        return meshes

    def tune_kernel(self, meshes):
        # select varifold kernel
        self.timer.start("tuning")

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
        self.timer.stop("tuning")

        return metric

    def compute_dist(self, meshes, metric):
        # flatten
        self.timer.start("dist")

        meshes_flat = ppdict.UnnestDict(sep="-")(meshes)
        self.results_["keys"] = list(meshes_flat.keys())

        # TODO: add tqdm
        dists = putils.pairwise_dists(list(meshes_flat.values()), metric.dist)

        self.timer.stop("dist")

        return dists

    def write(self):
        self.results_["time"] = self.timer.as_dict()

        with open(self.results_dir / "results.json", "w") as file:
            json.dump(self.results_, file, indent=4)

        # TODO: can dump only upper triangular?
        np.save(self.results_dir / "pair_dists.npy", self.dists_)

    def run(self):
        self.reset()

        self.timer.start("run")

        raw_meshes = self.load_data()
        meshes = self.preprocess_meshes(raw_meshes)
        metric = self.tune_kernel(meshes)
        self.dists_ = self.compute_dist(meshes, metric)

        self.timer.stop("run")

        self.write()

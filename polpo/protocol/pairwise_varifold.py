import json

import numpy as np

import polpo.preprocessing.dict as ppdict
import polpo.utils as putils
from polpo.mesh.surface import PvSurface
from polpo.mesh.varifold.tuning.metric_based import SigmaFromLengths
from polpo.preprocessing.mesh.registration import RigidAlignment
from polpo.time import Timer


class PairwiseVarifold:
    def __init__(self, known_correspondences, results_dir):
        self.timer = Timer()

        self.known_correspondences = known_correspondences
        self.results_dir = results_dir

        self.reset()

    def reset(self):
        self.results_ = {"version": "0.1.0"}
        self.dists_ = None
        self.timer.reset()

    def preprocess_meshes(self, raw_meshes):
        # rigidly aligns all the meshes against a randomly chosen target
        self.timer.start("prep")

        subj_key = putils.extract_random_key(raw_meshes)
        session_id = putils.extract_random_key(raw_meshes[subj_key])

        align_pipe = RigidAlignment(
            target=raw_meshes[subj_key][session_id],
            known_correspondences=self.known_correspondences,
        )

        meshes = (ppdict.DictMap(align_pipe + ppdict.DictMap(PvSurface)))(raw_meshes)

        self.timer.stop("prep")

        self.results_["rigid_alignment"] = {
            "subject_id": subj_key,
            "session_id": session_id,
            "known_correspondences": self.known_correspondences,
        }

        return meshes

    def tune_kernel(self, meshes):
        # select varifold kernel using a randomly selected mesh per subject
        self.timer.start("tuning")

        sigma_search = SigmaFromLengths(
            ratio_charlen_mesh=2.0,
            ratio_charlen=0.25,
        )

        mesh_per_subject = []
        subj_keys = {}
        for subj_key, subj_meshes in meshes.items():
            session_id = putils.extract_random_key(subj_meshes)
            mesh_per_subject.append(subj_meshes[session_id])

            subj_keys[subj_key] = session_id

        sigma_search.fit(mesh_per_subject)

        metric = sigma_search.optimal_metric_
        self.timer.stop("tuning")

        self.results_["kernel_tuning"] = {
            "sigma": sigma_search.sigma_,
            "meshes": subj_keys,
        }

        return metric

    def compute_dist(self, meshes, metric):
        # flatten
        self.timer.start("dist")

        meshes_flat = putils.unnest_dict(meshes, sep="-")
        self.results_["keys"] = list(meshes_flat.keys())

        # TODO: add tqdm? or timing per mesh?
        dists = putils.pairwise_dists(list(meshes_flat.values()), metric.dist)

        self.timer.stop("dist")

        return dists

    def write(self):
        self.results_["time"] = self.timer.as_dict()

        with open(self.results_dir / "results.json", "w") as file:
            json.dump(self.results_, file, indent=4)

        # TODO: can dump only upper triangular?
        np.save(self.results_dir / "pair_dists.npy", self.dists_)

    def run(self, dataset):
        # dataset: subject, session

        self.reset()

        self.timer.start("run")

        meshes = self.preprocess_meshes(dataset)
        metric = self.tune_kernel(meshes)
        self.dists_ = self.compute_dist(meshes, metric)

        self.timer.stop("run")

        self.write()

import json

from polpo.dataset import NestedDataset
from polpo.protocol.mixin import PairwiseDistancesMixin, RigidAlignmentMixin
from polpo.surface_mesh.euclidean import EuclideanSurfaces
from polpo.time import Timer


class PairwiseEuclidean(RigidAlignmentMixin, PairwiseDistancesMixin):
    # TODO: can create PairwiseInCorrespondence
    def __init__(
        self,
        results_dir,
        n_jobs=1,
    ):
        self.timer = Timer()

        self.results_dir = results_dir

        self.known_correspondences = True
        self.n_jobs = n_jobs

        self.reset()

    def reset(self):
        self.results_ = {"version": "0.1.0"}
        self.params_ = {}
        self.timer.reset()

        self.dists_ = None

    def write(self):
        with open(self.results_dir / "params.json", "w") as file:
            json.dump(self.params_, file, indent=2)

        with open(self.results_dir / "results.json", "w") as file:
            json.dump(self.results_, file, indent=2)

        self.dists_.save(self.results_dir / "pairwise_dists")

        with open(self.results_dir / "time.json", "w") as file:
            json.dump(self.timer.as_dict(), file, indent=2)

    def instantiate_metric(self, nested_meshes):
        faces = nested_meshes.flatten().sample().values_list()[0].faces

        space = EuclideanSurfaces(faces)

        self.params_["metric"] = {
            "metric": "euclidean",
        }

        return space.metric

    def run(self, nested_meshes):
        # nested_meshes: dict or polpo.dataset.NestedDataset
        if isinstance(nested_meshes, dict):
            nested_meshes = NestedDataset(nested_meshes)

        self.reset()

        self.timer.start("run")
        nested_meshes = self.preprocess_meshes(nested_meshes.flatten()).nest()

        metric = self.instantiate_metric(nested_meshes)
        self.dists_ = self.compute_pairwise_dists(nested_meshes.flatten(), metric)

        self.timer.stop("run")

        self.write()

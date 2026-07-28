import polpo.utils as putils
from polpo.preprocessing.mesh.registration import RigidAlignment
from polpo.surface_mesh.core import PvSurface

# TODO: create Protocol metaclass: timer, results_, params_


class MeshPreprocessorMixin:
    # TODO: use composition instead of mixin
    def preprocess_meshes(self, data):
        # data : polpo.dataset.Dataset

        # rigidly aligns all the meshes against a randomly chosen target
        self.timer.start("prep")

        key = putils.extract_random_key(data)
        aligner = RigidAlignment(
            target=data[key],
            known_correspondences=self.known_correspondences,
        )

        data_ = data.transform(aligner).map_values(PvSurface)

        self.timer.stop("prep")

        self.params_["rigid_alignment"] = {
            "known_correspondences": self.known_correspondences,
        }
        self.results_["rigid_alignment"] = {
            "key": key,
        }

        return data_

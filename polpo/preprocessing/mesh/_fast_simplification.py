import fast_simplification
import numpy as np
from sklearn.neighbors import NearestNeighbors

from polpo.preprocessing.base import PreprocessingStep


class FastSimplificationDecimator(PreprocessingStep):
    def __init__(self, target_reduction=0.25):
        super().__init__()
        self.target_reduction = target_reduction
        self._nbrs = NearestNeighbors(n_neighbors=1)

    def apply(self, mesh):
        # TODO: make a check for colors?
        vertices, faces, colors = mesh

        vertices_, faces_ = fast_simplification.simplify(
            vertices, faces, self.target_reduction
        )

        # TODO: can this be done better?
        self._nbrs.fit(vertices)
        _, indices = self._nbrs.kneighbors(vertices_)
        indices = np.squeeze(indices)
        colors_ = colors[indices]

        return vertices_, faces_, colors_

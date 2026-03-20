# artificial split of module to avoid import errors


import geomstats.backend as gs
import numpy as np

from polpo.mesh.qoi import centroid2farthest_vertex


class SigmaFromLengths:
    def __init__(
        self,
        ratio_charlen_mesh=2.0,
        ratio_charlen=0.25,
        charlen_fun=None,
    ):
        if charlen_fun is None:
            charlen_fun = centroid2farthest_vertex

        self.charlen_fun = charlen_fun
        self.ratio_charlen_mesh = ratio_charlen_mesh
        self.ratio_charlen = ratio_charlen

        self.sigma_ = None

    def fit(self, surfaces):
        charlen = self.charlen_fun(surfaces)
        charlen_mesh = gs.array(
            [np.median(surface.edge_lengths) for surface in surfaces]
        )

        sigmas = np.maximum(
            charlen * self.ratio_charlen,
            charlen_mesh * self.ratio_charlen_mesh,
        )

        # TODO: think about this
        self.sigma_ = np.amax(sigmas)

        return self

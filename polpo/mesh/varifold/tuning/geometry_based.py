# artificial split of module to avoid import errors


import geomstats.backend as gs

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
            [gs.median(surface.edge_lengths) for surface in surfaces]
        )

        sigmas = gs.maximum(
            charlen * self.ratio_charlen,
            charlen_mesh * self.ratio_charlen_mesh,
        )

        # TODO: think about this
        self.sigma_ = gs.amax(sigmas)

        return self

import abc
import itertools
import warnings

import geomstats.backend as gs
import numpy as np
from geomstats.varifold import (
    BinetKernel,
    GaussianKernel,
    SurfacesKernel,
    VarifoldMetric,
)

from polpo.mesh.geometry import centroid2farthest_vertex
from polpo.mesh.surface import PvSurface
from polpo.preprocessing import BranchingPipeline, Map


def _default_decimator():
    # TODO: pass float?
    from polpo.preprocessing.mesh.decimation import PvDecimate

    return BranchingPipeline(
        [
            PvDecimate(target_reduction=target_reduction, keep_colors=False)
            # TODO: update here
            for target_reduction in (0.1,)
        ],
        merger=lambda x: x,
    ) + Map(PvSurface)


class KernelFromSigma:
    def __init__(self, PositionKernel=None, TangentKernel=None):
        if PositionKernel is None:
            PositionKernel = GaussianKernel

        if TangentKernel is None:
            TangentKernel = BinetKernel

        self.PositionKernel = PositionKernel
        self.TangentKernel = TangentKernel

    def __call__(self, sigma):
        position_kernel = self.PositionKernel(sigma=sigma, init_index=0)

        tangent_kernel = self.TangentKernel(
            init_index=position_kernel.new_variable_index()
        )
        return SurfacesKernel(
            position_kernel,
            tangent_kernel,
        )


class GridFromMaxDist(abc.ABC):
    def __init__(self, ratios, distance=None):
        if distance is None:
            distance = centroid2farthest_vertex

        self.ratios = ratios
        self.distance = distance

    @classmethod
    def from_linspace(cls, min_ratio=0.05, max_ratio=1.0, grid_size=10):
        ratios = gs.linspace(min_ratio, max_ratio, num=grid_size)
        return cls(ratios)

    def __call__(self, surfaces):
        return self.ratios * gs.amax(self.distance(surfaces))


class _SigmaSearch(abc.ABC):
    def __init__(self, kernel_builder=None):
        if kernel_builder is None:
            kernel_builder = KernelFromSigma()

        self.kernel_builder = kernel_builder

        self.sigma_ = None

    @property
    def optimal_metric_(self):
        kernel = self.kernel_builder(self.sigma_)

        return VarifoldMetric(kernel)


class _DecimationBasedSigmaSearch(_SigmaSearch, abc.ABC):
    def __init__(self, kernel_builder=None, decimator=None):
        super().__init__(kernel_builder)

        if decimator is True:
            decimator = _default_decimator()

        self.decimator = decimator

        self.grid_ = None
        self.sdists_ = None

    @property
    def sigma_(self):
        return self.grid_[self.idx_]

    @sigma_.setter
    def sigma_(self, value):
        pass

    @property
    def sdist_(self):
        return self.sdists_[self.idx_]

    def _compute_dists_given_sigma(self, sigma, surface_pairs):
        kernel = self.kernel_builder(sigma)
        metric = VarifoldMetric(kernel)
        return gs.asarray([metric.squared_dist(*pair) for pair in surface_pairs])

    def _handle_decimation(self, surfaces):
        if self.decimator is None:
            return surfaces

        decimated_meshes = [self.decimator(surface) for surface in surfaces]

        surface_pairs = []
        for surface, decimated_meshes_ in zip(surfaces, decimated_meshes):
            surface_pairs.extend(
                list(itertools.combinations([surface] + decimated_meshes_, 2))
            )

        return surface_pairs


class SigmaGridSearch(_DecimationBasedSigmaSearch):
    """Grid search for best sigma.

    Notes
    -----
    * optimal value will be very dependent on the grid, prefer ``SigmaBisecSearch``.
    """

    def __init__(
        self,
        kernel_builder=None,
        elbow_finder=None,
        grid_generator=None,
        elbow_offset=1,
        decimator=None,
    ):
        super().__init__(kernel_builder=kernel_builder, decimator=decimator)

        if elbow_finder is None:
            from polpo.elbow import Rotor

            elbow_finder = Rotor()

        if grid_generator is None:
            grid_generator = GridFromMaxDist.from_linspace()

        self.elbow_finder = elbow_finder
        self.grid_generator = grid_generator
        self.elbow_offset = elbow_offset

    @property
    def idx_(self):
        return min(self.elbow_idx_ + self.elbow_offset, len(self.grid_) - 1)

    @property
    def elbow_idx_(self):
        return self.elbow_finder.elbow_index_

    def fit(self, surfaces):
        # assumes different parameterizations of same surface
        # list[(Surface, Surface)] or list[Surface]

        surface_pairs = self._handle_decimation(surfaces)

        self.grid_ = self.grid_generator(
            [surface_pair[0] for surface_pair in surface_pairs]
        )

        sdists = []
        for sigma in self.grid_:
            sdists.append(gs.sum(self._compute_dists_given_sigma(sigma, surface_pairs)))

        self.elbow_finder.fit(self.grid_, sdists)

        self.sdists_ = gs.array(sdists)

        return self


class SigmaBisecSearch(_DecimationBasedSigmaSearch):
    """Sigma bisection search.

    This method depends on the characteristic length of the mesh
    and should probably be avoided
    """

    def __init__(
        self,
        ref_value=0.1,
        atol=1e-4,
        kernel_builder=None,
        decimator=None,
        sigma_picker=None,
        max_iter=10,
    ):
        super().__init__(kernel_builder=kernel_builder, decimator=decimator)

        if sigma_picker is None:
            sigma_picker = SigmaPicker()

        self.ref_value = ref_value
        self.atol = atol
        self.sigma_picker = sigma_picker
        self.max_iter = max_iter

        self.idx_ = -1
        self.converged_ = None

    def _merge_dists(self, dists):
        return gs.mean(dists)

    def _append_res(self, sigma, sdist):
        self.grid_.append(sigma)
        self.sdists_.append(sdist)

    def _restart(self):
        self.converged_ = False
        self.grid_ = []
        self.sdists_ = []

    def fit(self, surfaces):
        # list[(Surface, Surface)] or list[Surface]
        self._restart()

        surface_pairs = self._handle_decimation(surfaces)

        sigma_a, sigma_b = self.sigma_picker.init(
            [surface_pair[0] for surface_pair in surface_pairs]
        )

        sdist_a = self._merge_dists(
            self._compute_dists_given_sigma(sigma_a, surface_pairs)
        )
        sdist_b = self._merge_dists(
            self._compute_dists_given_sigma(sigma_b, surface_pairs)
        )

        self._append_res(sigma_a, sdist_a)
        self._append_res(sigma_b, sdist_b)

        diff_b = sdist_b - self.ref_value  # expected to be negative
        if diff_b > 0.0:
            warnings.warn("No convergence. Upper interval bound is too strict.")
            return self

        diff_a = sdist_a - self.ref_value  # expected to be positive
        if diff_a < 0.0:
            self.converged_ = True
            warnings.warn("Converged in the lower bound. You can be less strict.")
            return self

        for _ in range(self.max_iter):
            sigma = self.sigma_picker.pick()
            sdist = self._merge_dists(
                self._compute_dists_given_sigma(sigma, surface_pairs)
            )

            self._append_res(sigma, sdist)

            signed_diff = sdist - self.ref_value
            stop = self.sigma_picker.update_bounds(sigma, signed_diff)
            if stop or abs(signed_diff) < self.atol:
                self.converged_ = True
                break
        else:
            warnings.warn("Maximum iterations reached.")


class SigmaPicker:
    def __init__(self, min_ratio=0.05, max_ratio=1.0, atol=None, distance=None):
        if distance is None:
            distance = centroid2farthest_vertex

        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.atol = atol
        self.distance = distance

        self.bounds = None
        self.bounds_ = None
        self.atol_ = None

    def init(self, surfaces):
        dists = self.distance(surfaces)

        ref_dist = gs.mean(dists)
        self.bounds = [
            self.min_ratio * ref_dist,
            self.max_ratio * ref_dist,
        ]
        self.bounds_ = self.bounds.copy()

        self.atol_ = self.atol
        if self.atol is None:
            a, b = self.bounds
            self.atol_ = (a + b) / 100.0

        return self.bounds

    def pick(self):
        a, b = self.bounds_
        return (a + b) / 2

    def update_bounds(self, sigma, diff):
        if diff < 0.0:
            self.bounds_[-1] = sigma
        else:
            self.bounds_[0] = sigma

        if self.bounds_[1] - self.bounds_[0] < self.atol_:
            return True

        return False


class SigmaFromLengths(_SigmaSearch):
    def __init__(
        self,
        ratio_charlen_mesh=2.0,
        ratio_charlen=0.25,
        charlen_fun=None,
        kernel_builder=None,
    ):
        super().__init__(kernel_builder)

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

        self.sigma_ = np.amax(sigmas)

        return self

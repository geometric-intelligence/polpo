import itertools

import geomstats.backend as gs
import numpy as np
from geomstats.varifold import (
    BinetKernel,
    GaussianKernel,
    SurfacesKernel,
    VarifoldMetric,
)

from polpo.elbow import Rotor


class SigmaBasedKernelBuilder:
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


class BoundingSphereBasedGridGenerator:
    def __init__(self, ratios):
        self.ratios = ratios

    @classmethod
    def from_linspace(cls, min_ratio=0.05, max_ratio=0.20, grid_size=5):
        ratios = gs.linspace(min_ratio, max_ratio, num=grid_size)
        return cls(ratios)

    def __call__(self, surface):
        # Trimesh or list[Trimesh]
        if isinstance(surface, (list, tuple)):
            surface = surface[0]

        max_dist = np.linalg.norm(surface.bounding_sphere.bounds)
        return self.ratios * max_dist


class GridBasedSigmaFinder:
    def __init__(self, kernel_builder=None, elbow_finder=None, grid_generator=None):
        if kernel_builder is None:
            kernel_builder = SigmaBasedKernelBuilder()

        if elbow_finder is None:
            elbow_finder = Rotor()

        if grid_generator is None:
            grid_generator = BoundingSphereBasedGridGenerator.from_linspace()

        self.kernel_builder = kernel_builder
        self.elbow_finder = elbow_finder
        self.grid_generator = grid_generator

        self.grid_ = None
        self.sdists_ = None

    @property
    def sigma_(self):
        return self.grid_[min(self.elbow_idx_ + 1, len(self.grid_) - 1)]

    @property
    def elbow_idx_(self):
        return self.elbow_finder.elbow_index_

    def fit(self, surfaces):
        # assumes different parameterizations of same surface
        # must be compatible with GridGenerator and KernelBuilder
        # list[Surface]
        self.grid_ = self.grid_generator(surfaces)

        surface_pairs = list(itertools.combinations(surfaces, 2))

        sdists = []
        for sigma in self.grid_:
            kernel = self.kernel_builder(sigma)

            metric = VarifoldMetric(kernel)

            sdists.append(
                gs.sum(gs.array([metric.squared_dist(*pair) for pair in surface_pairs]))
            )

        self.elbow_finder.fit(self.grid_, sdists)

        self.sdists_ = np.array(sdists)

        return self

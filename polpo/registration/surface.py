import time

import geomstats.backend as gs
from geomstats.geometry.discrete_surfaces import RelaxedPathStraightening, Surface
from geomstats.numerics.optimization import ScipyMinimize
from geomstats.numerics.path import UniformlySampledDiscretePath

from polpo.logging import logger


def _warn_max_iterations(iteration, max_iter):
    if iteration + 1 == max_iter:
        logger.warning(
            f"Maximum number of iterations {max_iter} reached. "
            "The estimate may be inaccurate"
        )


class LambdaAdaptiveRelaxedPathStraightening:
    def __init__(
        self,
        total_space,
        # TODO: improve, also in geomstats
        varifold_metric,
        initial_lambda=10.0,
        lambda_ratio=10.0,
        max_iter=5,
        sdist_tol=0.001,  # use distance to decimate
        optimizer=None,
    ):
        if optimizer is None:
            # TODO: update it in geomstats
            optimizer = ScipyMinimize(
                method="L-BFGS-B",
                autodiff_jac=True,
                options={"disp": False, "maxiter": 1000},
                tol=1e-4,
            )

        self.initial_lambda = initial_lambda
        self.lambda_ratio = lambda_ratio
        self.max_iter = max_iter
        self.sdist_tol = sdist_tol
        self.varifold_metric = varifold_metric

        self._straightener = RelaxedPathStraightening(
            total_space,
            # TODO: allow user to pass? may also pass list?
            n_nodes=3,
            optimizer=optimizer,
            discrepancy_loss=lambda point: varifold_metric.loss(
                point, target_faces=total_space.faces
            ),
        )

    def _new_initialization(self, discrete_path):
        geod_path = UniformlySampledDiscretePath(discrete_path, point_ndim=2)
        return lambda *args: geod_path(
            gs.linspace(0.0, 1.0, num=self._straightener.n_nodes)[1:]
        )

    def align(self, point, base_point, return_paths=False):
        self._straightener.lambda_ = self.initial_lambda
        self._straightener.initialization = self._straightener._default_initialization

        geods = []
        for iteration in range(self.max_iter):
            # TODO: control verbosity
            print("=======")
            print(f"Running algo {iteration}, lambda: {self._straightener.lambda_}")
            start_time = time.perf_counter()
            if geods:
                self._straightener.initialization = self._new_initialization(geods[-1])

            discrete_path = self._straightener.discrete_geodesic_bvp(point, base_point)
            geods.append(discrete_path)

            print(f"Algo {iteration} run in {time.perf_counter() - start_time:.2f} s")

            # TODO: may use a different strategy here if notion of discrepancy loss
            sdist = self.varifold_metric.squared_dist(
                point,
                Surface(
                    vertices=discrete_path[-1],
                    faces=self._straightener._total_space.faces,
                ),
            )
            if sdist < self.sdist_tol:
                break

            # TODO: update lambda
            self._straightener.lambda_ *= self.lambda_ratio

        else:
            _warn_max_iterations(iteration, self.max_iter)

        if return_paths:
            return geods

        return geods[-1][-1]

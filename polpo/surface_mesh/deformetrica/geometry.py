import shutil
from pathlib import Path

import numpy as np

import polpo.deformetrica as pdefo

from .core import RegistrationResult, ShootResult, TransportResult
from .paths import LddmmPaths


class LddmmMetric:
    def __init__(
        self,
        dir_config,
        kernel_width=10.0,
        recompute=False,
        use_pole_ladder=False,
        transport_zero_tol=1e-3,
        **registration_kwargs,
    ):
        if isinstance(dir_config, Path):
            dir_config = LddmmPaths(dir_config)

        self.dir_config = dir_config

        self.kernel_width = kernel_width
        self.use_pole_ladder = use_pole_ladder
        self.transport_zero_tol = transport_zero_tol
        self.registration_kwargs = registration_kwargs

        # TODO: cache_policy: reuse, overwrite, validate, read_only
        self.recompute = recompute

        deformation_kernel = pdefo.geometry.kernel_factory.factory(
            kernel_type="torch",
            kernel_width=kernel_width,
        )
        self._exponential = pdefo.geometry.Exponential(kernel=deformation_kernel)

    def _dir_exists(self, dirname):
        if self.recompute and dirname.exists():
            shutil.rmtree(dirname)

        return dirname.exists()

    def log(self, point, base_point):
        # TODO: make _single and vectorize?

        id_ = f"{base_point.id}_to_{point.id}"
        dir_ = RegistrationResult(id_, self.dir_config, base_point, point)

        # TODO: make this part of RegistrationDir?
        if not self._dir_exists(dir_.dirname):
            pdefo.registration.estimate_registration(
                base_point.as_vtk_path(),
                point.as_vtk_path(),
                target_id=point.id,
                output_dir=dir_.dirname,
                kernel_width=self.kernel_width,
                **self.registration_kwargs,
            )
            dir_.write()

        # TODO: if exists, check if other meshes are being used?

        return dir_.tangent_vec

    def exp(self, tangent_vec, base_point):
        dir_ = ShootResult(
            f"{base_point.id}_shoot_{tangent_vec.id}",
            self.dir_config,
            tangent_vec,
            base_point,
        )

        if not self._dir_exists(dir_.dirname):
            pdefo.geometry.shoot(
                source=base_point.as_vtk_path(),
                control_points=tangent_vec.control_points.as_path(),
                momenta=tangent_vec.momenta.as_path(),
                kernel_width=self.kernel_width,
                # TODO: add shoot params?
                concentration_of_time_points=10,
                kernel_type="torch",
                output_dir=dir_.dirname,
                # TODO: control it at init?
                # TODO: compare geodesic with parallel transport fan one
                write_adjoint_parameters=False,
            )
            dir_.write()

        return dir_.point

    def parallel_transport(
        self, tangent_vec, base_point, direction=None, end_point=None
    ):
        if direction is None:
            # TODO: implement? it is actually easy
            raise NotImplementedError("Need direction to compute parallel transport")

        method = "pole_ladder" if self.use_pole_ladder else "fan"
        if (
            self.transport_zero_tol is not None
            and self.squared_norm(tangent_vec) < self.transport_zero_tol**2
        ):
            method = "zero"

        dir_ = TransportResult(
            f"{tangent_vec.id}_along_{direction.id}",
            self.dir_config,
            tangent_vec,
            base_point,
            direction,
            method=method,
        )

        # TODO: control at init?
        if not self._dir_exists(dir_.dirname):
            if method != "zero":
                pdefo.geometry.parallel_transport(
                    source=base_point.as_vtk_path(),
                    control_points=direction.control_points.as_path(),
                    momenta=direction.momenta.as_path(),
                    control_points_to_transport=tangent_vec.control_points.as_path(),
                    momenta_to_transport=tangent_vec.momenta.as_path(),
                    kernel_width=self.kernel_width,
                    output_dir=dir_.dirname,
                    use_pole_ladder=self.use_pole_ladder,  # TODO: just use a different scheme?
                )
            else:
                dir_.write_data()

            dir_.write()

        return dir_.transported

    def squared_norm(self, tangent_vec, base_point=None):
        # NB: base_point is ignored
        control_points, momenta = pdefo.utils.move_data(
            tangent_vec.control_points.as_array(),
            tangent_vec.momenta.as_array(),
        )

        return self._exponential.scalar_product(
            control_points,
            momenta,
            momenta,
        )

    def norm(self, tangent_vec, base_point=None):
        return np.sqrt(self.squared_norm(tangent_vec, base_point).numpy())

import shutil
from pathlib import Path

import polpo.deformetrica as pdefo

from .core import (
    RegistrationDir,
    ShootDir,
    TransportDir,
)
from .utils import DirConfig


class LddmmMetric:
    def __init__(
        self,
        dir_config,
        kernel_width=10.0,
        recompute=False,
        use_pole_ladder=False,
        **registration_kwargs,
    ):
        if isinstance(dir_config, Path):
            dir_config = DirConfig(dir_config)
        self.dir_config = dir_config

        self.kernel_width = kernel_width
        self.use_pole_ladder = use_pole_ladder
        self.registration_kwargs = registration_kwargs

        # TODO: cache_policy: reuse, overwrite, validate, read_only
        self.recompute = recompute

        # TODO: create only when required?

    def _dir_exists(self, dirname):
        if self.recompute and dirname.exists():
            shutil.rmtree(dirname)

        return dirname.exists()

    def log(self, point, base_point):
        # TODO: make _single and vectorize?

        id_ = f"{base_point.id}_to_{point.id}"
        dir_ = RegistrationDir(
            self.dir_config.registration_dir / id_,
            base_point,
            point,
        )

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
            dir_.write_json()

        # TODO: if exists, check if other meshes are being used?

        return dir_.tangent_vec()

    def exp(self, tangent_vec, base_point):
        dir_ = ShootDir(
            self.dir_config.shoot_dir / f"{base_point.id}_shoot_{tangent_vec.id}",
            tangent_vec,
            base_point,
        )

        if not self._dir_exists(dir_.dirname):
            pdefo.geometry.shoot(
                source=base_point.as_vtk_path(),
                control_points=tangent_vec.control_points().as_path(),
                momenta=tangent_vec.momenta().as_path(),
                kernel_width=self.kernel_width,
                # TODO: add shoot params?
                concentration_of_time_points=10,
                kernel_type="torch",
                output_dir=dir_.dirname,
                # TODO: control it at init?
                # TODO: compare geodesic with parallel transport fan one
                write_adjoint_parameters=False,
            )
            dir_.write_json()

        return dir_.point()

    def parallel_transport(
        self, tangent_vec, base_point, direction=None, end_point=None
    ):
        if direction is None:
            # TODO: implement? it is actually easy
            raise NotImplementedError("Need direction to compute parallel transport")

        scheme = "ladder" if self.use_pole_ladder else "fan"
        dir_ = TransportDir(
            self.dir_config.transport_dir
            / f"{tangent_vec.id}_along_{scheme}_{direction.id}",
            tangent_vec,
            base_point,
            direction,
            pole_ladder=self.use_pole_ladder,
        )

        # TODO: control at init?
        if not self._dir_exists(dir_.dirname):
            pdefo.geometry.parallel_transport(
                source=base_point.as_vtk_path(),
                control_points=direction.control_points().as_path(),
                momenta=direction.momenta().as_path(),
                control_points_to_transport=tangent_vec.control_points().as_path(),
                momenta_to_transport=tangent_vec.momenta().as_path(),
                kernel_width=self.kernel_width,
                output_dir=dir_.dirname,
                use_pole_ladder=self.use_pole_ladder,  # TODO: just use a different scheme?
            )
            dir_.write_json()

        return dir_.transported()

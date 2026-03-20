import shutil

from in_out.array_readers_and_writers import write_3D_array

import polpo.deformetrica as pdefo

from .repr import (
    DeterministicAtlasPoint,
    ShootedPoint,
    TangentVecFromRegistration,
    TransportedVec,
)

# TODO: put all deformetrica things in one place, and then play with imports


class LddmmMetric:
    def __init__(
        self,
        outputs_dir,
        meshes_dir=None,
        registration_dir=None,
        transport_dir=None,
        shoot_dir=None,
        atlas_dir=None,
        kernel_width=10.0,
        recompute=False,
        use_pole_ladder=False,
        **registration_kwargs,
    ):
        # TODO: create notion of dir configuration? and add functions there?

        if meshes_dir is None:
            meshes_dir = outputs_dir / "meshes"

        if registration_dir is None:
            registration_dir = outputs_dir / "registrations"

        if transport_dir is None:
            transport_dir = outputs_dir / "transports"

        if shoot_dir is None:
            shoot_dir = outputs_dir / "shoots"

        if atlas_dir is None:
            atlas_dir = outputs_dir / "atlases"

        self.outputs_dir = outputs_dir
        self.meshes_dir = meshes_dir
        self.registration_dir = registration_dir
        self.shoot_dir = shoot_dir
        self.atlas_dir = atlas_dir
        self.transport_dir = transport_dir

        self.kernel_width = kernel_width
        self.use_pole_ladder = use_pole_ladder
        self.registration_kwargs = registration_kwargs

        self.recompute = recompute

        # TODO: create only when required?
        self.meshes_dir.mkdir(parents=True, exist_ok=True)

    def _dir_exists(self, dirname):
        if self.recompute and dirname.exists():
            shutil.rmtree(dirname)

        return dirname.exists()

    def all_dirs(self):
        return {
            "meshes": self.meshes_dir.relative_to(self.outputs_dir).as_posix(),
            "registrations": self.registration_dir.relative_to(
                self.outputs_dir
            ).as_posix(),
            "transports": self.transport_dir.relative_to(self.outputs_dir).as_posix(),
            "shoots": self.shoot_dir.relative_to(self.outputs_dir).as_posix(),
            "atlases": self.atlas_dir.relative_to(self.outputs_dir).as_posix(),
        }

    def log(self, point, base_point):
        # TODO: make _single and vectorize?

        vec = TangentVecFromRegistration(
            base_point, point, outputs_dir=self.registration_dir
        )

        if not self._dir_exists(vec.dirname):
            pdefo.registration.estimate_registration(
                base_point.as_vtk_path(),
                point.as_vtk_path(),
                target_id=point.id,
                output_dir=vec.dirname,
                kernel_width=self.kernel_width,
                **self.registration_kwargs,
            )

        return vec

    def exp(self, tangent_vec, base_point):
        point = ShootedPoint(
            base_point,
            tangent_vec,
            outputs_dir=self.shoot_dir,
        )

        if not self._dir_exists(point.dirname):
            pdefo.geometry.shoot(
                source=base_point.as_vtk_path(),
                control_points=tangent_vec.control_points(),
                momenta=tangent_vec.momenta(),
                kernel_width=self.kernel_width,
                # TODO: add shoot params?
                concentration_of_time_points=10,
                kernel_type="torch",
                output_dir=point.dirname,
                # TODO: control it at init?
                # TODO: compare geodesic with parallel transport fan one
                write_adjoint_parameters=False,
            )

        return point

    def parallel_transport(
        self, tangent_vec, base_point, direction=None, end_point=None
    ):
        if direction is None:
            # TODO: implement? it is actually easy
            raise NotImplementedError("Need direction to compute parallel transport")

        vec = TransportedVec(
            tangent_vec,
            base_point,
            direction,
            outputs_dir=self.transport_dir,
            pole_ladder=self.use_pole_ladder,
        )

        # TODO: control at init?
        if not self._dir_exists(vec.dirname):
            pdefo.geometry.parallel_transport(
                source=base_point.as_vtk_path(),
                control_points=direction.control_points(),
                momenta=direction.momenta(),
                control_points_to_transport=tangent_vec.control_points(),
                momenta_to_transport=tangent_vec.momenta(),
                kernel_width=self.kernel_width,
                output_dir=vec.dirname,
                use_pole_ladder=self.use_pole_ladder,  # TODO: just use a different scheme?
            )

        return vec


class FrechetMean:
    def __init__(self, metric, initial_step_size=1e-4, recompute=False):
        # TODO: space? seems overkill for goal
        self.metric = metric
        self.initial_step_size = initial_step_size

        self.recompute = recompute

        self.estimate_ = None

    def _dir_exists(self, dirname):
        if self.recompute and dirname.exists():
            shutil.rmtree(dirname)

        return dirname.exists()

    def fit(self, X, atlas_id):
        # TODO: need output dir
        # TODO: template
        self.estimate_ = None

        template = DeterministicAtlasPoint(
            atlas_id, points=X, outputs_dir=self.metric.atlas_dir
        )

        # TODO: might need to string it
        dataset = {point.id: point.as_vtk_path() for point in X}

        if not self._dir_exists(template.dirname):
            pdefo.learning.estimate_deterministic_atlas(
                targets=dataset,
                output_dir=template.dirname,
                initial_step_size=self.initial_step_size,
                kernel_width=self.metric.kernel_width,
                **self.metric.registration_kwargs,
            )

            momenta = pdefo.io.load_momenta(template.dirname, as_path=False)
            for momenta_, point in zip(momenta, X):
                filename = f"DeterministicAtlas__EstimatedParameters__Momenta__subject_{point.id}.txt"
                write_3D_array(momenta_, template.dirname, filename)

        self.estimate_ = template

        return self

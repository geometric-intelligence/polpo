"""A filesystem-backed adapter around deformetrica."""

from abc import ABC
from pathlib import Path

import numpy as np
import pyvista as pv

import polpo.deformetrica.io as pdefoio
from polpo.auto_all import auto_all
from polpo.io.json import load_json, save_json
from polpo.surface_mesh.core import PvSurface


class Point:
    def __init__(self, id_, pv_surface=None, vtk_path=None, dirname=None):
        self.id = id_
        self.pv_surface = pv_surface

        if vtk_path is None and dirname is None:
            raise ValueError("Need to define ``vtk_path`` or ``dirname``")

        if vtk_path is None:
            vtk_path = dirname / f"{self.id}.vtk"

        self.vtk_path = vtk_path

    def as_vtk_path(self):
        if self.vtk_path.exists():
            return self.vtk_path

        if self.pv_surface is None:
            raise ValueError("There's no mesh attached to this point.")

        self.vtk_path.parent.mkdir(parents=True, exist_ok=True)
        self.pv_surface.save(self.vtk_path)

        return self.vtk_path

    def as_polydata(self):
        if self.pv_surface is not None:
            return self.pv_surface

        if self.vtk_path is None:
            raise ValueError("There's no mesh attached to this point")

        self.pv_surface = pv.read(self.vtk_path)

        return self.pv_surface

    def as_pv_surface(self):
        return PvSurface(self.as_polydata(), id_=self.id)

    def to_dict(self, *, root_dir=None):
        vtk_path = self.vtk_path

        if root_dir is not None:
            vtk_path = vtk_path.relative_to(root_dir)

        return {
            "id": self.id,
            "vtk_path": vtk_path.as_posix(),
        }

    @classmethod
    def from_dict(cls, data, root_dir=None):
        vtk_path = Path(data["vtk_path"])

        if root_dir is not None and not vtk_path.is_absolute():
            vtk_path = root_dir / vtk_path

        return cls(
            data["id"],
            vtk_path=vtk_path,
        )


class ControlPoints:
    def __init__(self, filename):
        self.filename = filename

    def as_path(self):
        return self.filename

    def as_array(self):
        return pdefoio.read_array(self.filename)

    def as_pv(self):
        return pv.PolyData(self.as_array())


class Momenta:
    # TODO: homogenize with control points?
    def __init__(self, filename):
        self.filename = filename

    def as_path(self):
        return self.filename

    def as_array(self):
        return pdefoio.read_array(self.filename)

    def as_pv(self):
        # TODO: implement
        pass


class TangentVector:
    def __init__(self, id_, dirname):
        # TODO: allow id to be none?
        self.id = id_
        self.dirname = dirname

    def control_points(self):
        return ControlPoints(pdefoio.load_cp(self.dirname, as_path=True))

    def momenta(self):
        try:
            filename = pdefoio.load_momenta(self.dirname, as_path=True)
        except FileNotFoundError:
            filename = pdefoio.load_deterministic_atlas_momenta(
                self.dirname, as_path=True, id_=self.id.split("_to_")[-1]
            )

        return Momenta(filename)

    def to_dict(self, root_dir=None):
        dirname = self.dirname
        if root_dir is not None:
            dirname = dirname.relative_to(root_dir)

        return dict(id=self.id, dirname=dirname.as_posix())

    @classmethod
    def from_dict(cls, data, root_dir=None):
        dirname = Path(data["dirname"])

        if root_dir is not None and not dirname.is_absolute():
            dirname = Path(root_dir) / dirname

        return cls(id_=data["id"], dirname=dirname)


class TransportedVector(TangentVector):
    def control_points(self):
        return ControlPoints(pdefoio.load_transported_cp(self.dirname, as_path=True))

    def momenta(self):
        return Momenta(pdefoio.load_transported_momenta(self.dirname, as_path=True))


class Flow:
    def __init__(self, points, times=None):
        if times is None:
            times = np.linspace(0.0, 1.0, len(points))

        self.points = points
        self.times = times

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        return self.points[index]

    @property
    def initial_point(self):
        return self.points[0]

    @property
    def end_point(self):
        return self.points[0]

    def as_polydata(self):
        return [point.as_polydata() for point in self.points]

    def nearest(self, time):
        index = np.argmin(np.abs(self.times - time))
        return self.points[index]

    def at_sampled_time(self, time):
        indices = np.flatnonzero(np.isclose(self.times, time))
        if len(indices) == 0:
            raise ValueError(f"No flow point sampled at time {time}")
        return self.points[indices[0]]


class _Result(ABC):
    def write(self):
        return save_json(self.dirname / "params.json", self.params())


class RegistrationResult(_Result):
    # TODO: check success of registration; write to params?
    def __init__(self, id_, dir_config, base_point, point):
        self.id = id_
        self.dir_config = dir_config

        self.base_point = base_point
        self.point = point

    @property
    def dirname(self):
        return self.dir_config.registration_path(self.id)

    @classmethod
    def load(cls, id_, dir_config):
        data = load_json(dir_config.registration_path(id_) / "params.json")

        point = Point.from_dict(
            data["point"],
            root_dir=dir_config.outputs_dir,
        )
        base_point = Point.from_dict(
            data["base_point"],
            root_dir=dir_config.outputs_dir,
        )

        return cls(id_, dir_config, base_point, point)

    def params(self):
        root_dir = self.dir_config.outputs_dir

        return dict(
            base_point=self.base_point.to_dict(root_dir=root_dir),
            point=self.point.to_dict(root_dir=root_dir),
        )

    def tangent_vec(self):
        return TangentVector(self.id, self.dirname)

    def reconstructed(self):
        # TODO: same for template?

        vkt_path = pdefoio.load_deterministic_atlas_reconstruction(
            self.dirname, as_path=True, id_=self.point.id
        )
        return Point(
            id_=f"{self.base_point.id}_shoot_{self.dirname.name}", vtk_path=vkt_path
        )

    def flow(self):
        vtk_paths = pdefoio.load_deterministic_atlas_flow(
            self.dirname, as_pv=True, as_path=True
        )
        return Flow(
            [
                Point(f"{self.dirname.name}|tp{index}", vtk_path=vtk_path)
                for index, vtk_path in enumerate(vtk_paths)
            ]
        )


class ShootResult(_Result):
    def __init__(self, id_, dir_config, tangent_vec, base_point):
        self.id = id_
        self.dir_config = dir_config

        self.tangent_vec = tangent_vec
        self.base_point = base_point

    @property
    def dirname(self):
        return self.dir_config.shoot_path(self.id)

    @classmethod
    def load(cls, id_, dir_config):
        data = load_json(dir_config.shoot_path(id_) / "params.json")

        tangent_data = data["tangent_vec"]
        if Path(tangent_data["dirname"]).is_relative_to(
            dir_config.transports_dir.relative_to(dir_config.outputs_dir)
        ):
            tangent_vec = TransportedVector.from_dict(
                tangent_data,
                root_dir=dir_config.outputs_dir,
            )
        else:
            tangent_vec = TangentVector.from_dict(
                tangent_data,
                root_dir=dir_config.outputs_dir,
            )

        base_point = Point.from_dict(
            data["base_point"], root_dir=dir_config.outputs_dir
        )

        return cls(id_, dir_config, tangent_vec, base_point)

    def params(self):
        root_dir = self.dir_config.outputs_dir

        return dict(
            tangent_vec=self.tangent_vec.to_dict(root_dir=root_dir),
            base_point=self.base_point.to_dict(root_dir=root_dir),
        )

    def point(self):
        return Point(
            self.dirname.name,
            vtk_path=pdefoio.load_shooted_point(self.dirname, as_path=True),
        )

    def flow(self):
        vtk_paths = pdefoio.load_shooting_flow(
            self.dirname,
            as_pv=True,
            as_path=True,
        )
        return Flow(
            [
                Point(f"{self.dirname.name}|tp{index}", vtk_path=vtk_path)
                for index, vtk_path in enumerate(vtk_paths)
            ]
        )


class _BaseDeterministicAtlasResult(_Result):
    def __init__(self, id_, dir_config, points):
        self.id = id_
        self.dir_config = dir_config
        self.points = points

    @property
    def dirname(self):
        return self.dir_config.atlas_path(self.id)

    @classmethod
    def load(cls, id_, dir_config):
        data = load_json(dir_config.atlas_path(id_) / "params.json")

        points = [
            Point.from_dict(data_, root_dir=dir_config.outputs_dir)
            for data_ in data["points"]
        ]
        return cls(id_, dir_config, points)

    def params(self):
        return dict(
            points=[
                pt.to_dict(root_dir=self.dir_config.outputs_dir) for pt in self.points
            ]
        )


class DeterministicAtlasManyResult(_BaseDeterministicAtlasResult):
    # TODO: add to_registrations

    def template(self):
        return Point(
            self.id,
            vtk_path=pdefoio.load_template(self.dirname, as_path=True),
        )

    def control_points(self):
        # shared for all tangent vectors
        return ControlPoints(pdefoio.load_cp(self.dirname, as_path=True))

    def tangent_vecs(self):
        return [
            TangentVector(f"{self.id}_to_{pt.id}", self.dirname) for pt in self.points
        ]

    def flows(self):
        flows = {}
        for point in self.points:
            vtk_paths = pdefoio.load_deterministic_atlas_flow(
                self.dirname, as_path=True, id_=point.id
            )

            flows[point.id] = Flow(
                [
                    Point(f"{self.id}_to_{point.id}|tp{index}", vtk_path=vtk_path)
                    for index, vtk_path in enumerate(vtk_paths)
                ]
            )

        return flows

    def reconstructed(self):
        reconstructed = []
        for point in self.points:
            vkt_path = pdefoio.load_deterministic_atlas_reconstruction(
                self.dirname, as_path=True, id_=point.id
            )
            reconstructed.append(
                Point(id_=f"{self.id}_shoot_{self.id}_to_{point.id}", vtk_path=vkt_path)
            )

        return reconstructed


class DeterministicAtlasOneDir(_BaseDeterministicAtlasResult):
    def template(self):
        return Point(
            self.id,
            vtk_path=self.dirname / f"{self.id}.vtk",
        )

    def reconstructed(self):
        return [self.template()]

    def write_mesh(self):
        self.dirname.mkdir(parents=True)
        point = self.points[0]
        point.as_polydata().save(self.template().vtk_path)


class DeterministicAtlasResult(_BaseDeterministicAtlasResult):
    def __new__(cls, id_, dir_config, points):
        if len(points) == 1:
            return DeterministicAtlasOneDir(id_, dir_config, points)

        return DeterministicAtlasManyResult(id_, dir_config, points)


class _TransportResult(_Result):
    def __init__(self, id_, dir_config, tangent_vec, base_point, direction):
        # TODO: play with end_point and direction
        self.id = id_
        self.dir_config = dir_config

        self.tangent_vec = tangent_vec
        self.base_point = base_point
        self.direction = direction

    @property
    def dirname(self):
        return self.dir_config.transport_path(self.id)

    def params(self):
        root_dir = self.dir_config.outputs_dir

        return dict(
            tangent_vec=self.tangent_vec.to_dict(root_dir=root_dir),
            base_point=self.base_point.to_dict(root_dir=root_dir),
            direction=self.direction.to_dict(root_dir=root_dir),
            pole_ladder=not isinstance(self, TransportResultFan),
        )

    def transported(self):
        return TransportedVector(self.dirname.name, self.dirname)


class TransportResultFan(_TransportResult):
    def reconstructed(self):
        # TODO: update
        # NB: it reconstructs the end point of direction

        # TODO: control dirname? or maybe only point?
        # TODO: add flow? (not relevant right now)
        id_ = f"{self.direction.id}_r"
        vtk_path = pdefoio.load_shooted_point(self.dirname, as_path=True)
        return Point(id_=id_, vtk_path=vtk_path)

    def reconstructed_shooted(self):
        # TODO: update
        id_ = f"{self.direction.id}_rs"
        vtk_path = pdefoio.load_parallel_shooted_point(self.dirname, as_path=True)
        return Point(id_=id_, vtk_path=vtk_path)


class TransportResult:
    def __new__(cls, *args, pole_ladder=True, **kwargs):
        if pole_ladder:
            return _TransportResult(*args, **kwargs)

        return TransportResultFan(*args, **kwargs)

    @classmethod
    def load(cls, id_, dir_config):
        data = load_json(dir_config.transport_path(id_) / "params.json")

        tangent_vec = TangentVector.from_dict(
            data["tangent_vec"],
            root_dir=dir_config.outputs_dir,
        )
        base_point = Point.from_dict(
            data["base_point"],
            root_dir=dir_config.outputs_dir,
        )
        direction = TangentVector.from_dict(
            data["direction"],
            root_dir=dir_config.outputs_dir,
        )

        return cls(
            id_,
            dir_config,
            tangent_vec,
            base_point,
            direction,
            pole_ladder=data["pole_ladder"],
        )


__all__ = auto_all(globals())

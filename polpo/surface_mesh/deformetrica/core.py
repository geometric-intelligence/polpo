"""A filesystem-backed adapter around deformetrica."""

import json

import pyvista as pv

import polpo.deformetrica.io as pdefoio
from polpo.auto_all import auto_all
from polpo.surface_mesh.core import PvSurface


class Point:
    def __init__(self, id_, pv_surface=None, vtk_path=None, dirname=None):
        self.id = id_
        # TODO: rename?
        self.pv_surface = pv_surface

        if vtk_path is None and dirname is None:
            raise ValueError("Need to define ``vtk_path`` or ``dirname``")

        if vtk_path is None:
            vtk_path = dirname / f"{self.id}.vtk"

        self._vtk_path = vtk_path

    @property
    def vtk_path(self):
        return self._vtk_path

    def as_vtk_path(self):
        if self.vtk_path.exists():
            return self.vtk_path

        if self.pv_surface is None:
            raise ValueError("There's no mesh attached to this point.")

        self.pv_surface.save(self.vtk_path)

        return self.vtk_path

    def as_pv(self):
        if self.pv_surface is not None:
            return self.pv_surface

        if self.vtk_path is None:
            raise ValueError("There's no mesh attached to this point")

        self.pv_surface = pv.read(self.vtk_path)

        return self.pv_surface

    def as_pv_surface(self):
        return PvSurface(self.as_pv(), id_=self.id)

    def as_dict(self):
        return dict(id=self.id, vtk_path=self.vtk_path.as_posix())

    @classmethod
    def from_dict(cls, data):
        return cls(id_=data["id"], vtk_path=data["vtk_path"])


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

    def as_dict(self):
        return dict(id=self.id, dirname=self.dirname.as_posix())

    @classmethod
    def from_dict(cls, data):
        return cls(id_=data["id"], dirname=data["dirname"])


class TransportedVector(TangentVector):
    def control_points(self):
        return ControlPoints(pdefoio.load_transported_cp(self.dirname, as_path=True))

    def momenta(self):
        return Momenta(pdefoio.load_transported_momenta(self.dirname, as_path=True))


class RegistrationDir:
    # TODO: disambiguate template_shape: confirm it is source

    def __init__(self, dirname, base_point, point):
        self.dirname = dirname

        self.base_point = base_point
        self.point = point

    @classmethod
    def from_dirname(cls, dirname):
        # TODO: check if it exists

        # TODO: check if file exists?
        with open(dirname / "params.json", "r") as file:
            data = json.load(file)

        point = Point.from_dict(data["point"])
        base_point = Point.from_dict(data["base_point"])

        return cls(dirname, base_point, point)

    def params(self):
        return dict(base_point=self.base_point.as_dict(), point=self.point.as_dict())

    def write_json(self):
        with open(self.dirname / "params.json", "w") as file:
            json.dump(self.params(), file, indent=2)

    def tangent_vec(self):
        return TangentVector(self.dirname.name, self.dirname)

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
        return [
            Point(f"{self.dirname.name}|tp{index}", vtk_path=vtk_path)
            for index, vtk_path in enumerate(vtk_paths)
        ]


class ShootDir:
    # TODO: add Dir
    def __init__(self, dirname, tangent_vec, base_point):
        self.dirname = dirname

        self.tangent_vec = tangent_vec
        self.base_point = base_point

    @classmethod
    def from_dirname(cls, dirname):
        # TODO: check if it exists

        # TODO: check if file exists?
        with open(dirname / "params.json", "r") as file:
            data = json.load(file)

        tangent_vec = TangentVector.from_dict(data["tangent_vec"])
        base_point = Point.from_dict(data["base_point"])

        return cls(dirname, tangent_vec, base_point)

    def params(self):
        return dict(
            tangent_vec=self.tangent_vec.as_dict(),
            base_point=self.base_point.as_dict(),
        )

    def write_json(self):
        with open(self.dirname / "params.json", "w") as file:
            json.dump(self.params(), file, indent=2)

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
        return [
            Point(f"{self.dirname.name}|tp{index}", vtk_path=vtk_path)
            for index, vtk_path in enumerate(vtk_paths)
        ]


class _BaseDeterministicAtlasDir:
    def __init__(self, dirname, points):
        self.dirname = dirname
        self.points = points

    @classmethod
    def from_dirname(cls, dirname):
        # TODO: update

        # TODO: check if it exists

        # TODO: check if file exists?
        with open(dirname / "params.json", "r") as file:
            data = json.load(file)

        points = [Point.from_dict(data_) for data_ in data["points"]]
        return cls(dirname, points)

    def params(self):
        return dict(points=[pt.as_dict() for pt in self.points])

    def write_json(self):
        with open(self.dirname / "params.json", "w") as file:
            json.dump(self.params(), file, indent=2)


class DeterministicAtlasManyDir(_BaseDeterministicAtlasDir):
    # TODO: add to_registrations

    def template(self):
        return Point(
            self.dirname.name,
            vtk_path=pdefoio.load_template(self.dirname, as_path=True),
        )

    def control_points(self):
        # shared for all tangent vectors
        return ControlPoints(pdefoio.load_cp(self.dirname, as_path=True))

    def tangent_vecs(self):
        atlas_id = self.dirname.name
        return [
            TangentVector(f"{atlas_id}_to_{pt.id}", self.dirname) for pt in self.points
        ]

    def flows(self):
        atlas_id = self.dirname.name

        flows = []
        for point in self.points:
            vtk_paths = pdefoio.load_deterministic_atlas_flow(
                self.dirname, as_path=True, id_=point.id
            )

            flows.append(
                [
                    Point(f"{atlas_id}_to_{point.id}|tp{index}", vtk_path=vtk_path)
                    for index, vtk_path in enumerate(vtk_paths)
                ]
            )

        return flows

    def reconstructed(self):
        atlas_id = self.dirname.name

        reconstructed = []
        for point in self.points:
            vkt_path = pdefoio.load_deterministic_atlas_reconstruction(
                self.dirname, as_path=True, id_=point.id
            )
            reconstructed.append(
                Point(
                    id_=f"{atlas_id}_shoot_{atlas_id}_to_{point.id}", vtk_path=vkt_path
                )
            )

        return reconstructed


class DeterministicAtlasOneDir(_BaseDeterministicAtlasDir):
    def template(self):
        name = self.dirname.name
        return Point(
            name,
            vtk_path=self.dirname / f"{name}.vtk",
        )

    def reconstructed(self):
        return [self.template()]

    def write_mesh(self):
        self.dirname.mkdir(parents=True)
        point = self.points[0]
        point.as_pv().save(self.template().vtk_path)


class DeterministicAtlasDir(_BaseDeterministicAtlasDir):
    def __new__(cls, dirname, points):
        if len(points) == 1:
            return DeterministicAtlasOneDir(dirname, points)

        return DeterministicAtlasManyDir(dirname, points)


class _TransportDir:
    def __init__(self, dirname, tangent_vec, base_point, direction):
        # TODO: play with end_point and direction
        self.dirname = dirname

        self.tangent_vec = tangent_vec
        self.base_point = base_point
        self.direction = direction

    def params(self):
        return dict(
            tangent_vec=self.tangent_vec.as_dict(),
            base_point=self.base_point.as_dict(),
            direction=self.direction.as_dict(),
            pole_ladder=not isinstance(self, TransportDirFan),
        )

    def write_json(self):
        with open(self.dirname / "params.json", "w") as file:
            json.dump(self.params(), file, indent=2)

    def transported(self):
        return TransportedVector(self.dirname.name, self.dirname)


class TransportDirFan(_TransportDir):
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


class TransportDir:
    def __new__(cls, *args, pole_ladder=True, **kwargs):
        if pole_ladder:
            return _TransportDir(*args, **kwargs)

        return TransportDirFan(*args, **kwargs)

    @classmethod
    def from_dirname(cls, dirname):
        # TODO: check if it exists

        # TODO: check if file exists?
        with open(dirname / "params.json", "r") as file:
            data = json.load(file)

        tangent_vec = TangentVector.from_dict(data["tangent_vec"])
        base_point = Point.from_dict(data["base_point"])
        direction = TangentVector.from_dict(data["direction"])

        return cls(
            dirname, tangent_vec, base_point, direction, pole_ladder=data["pole_ladder"]
        )


__all__ = auto_all(globals())

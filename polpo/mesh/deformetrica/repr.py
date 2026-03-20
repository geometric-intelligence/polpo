import abc
from pathlib import Path

import pyvista as pv

import polpo.deformetrica.io as pdefoio
from polpo.auto_all import auto_all


class _Point(abc.ABC):
    # TODO: transform some of the variables into functions instead of properties?

    def __init__(self, id_, pv_surface=None):
        self.id = id_
        self.pv_surface = pv_surface

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

    @property
    @abc.abstractmethod
    def vtk_path(self):
        pass


class Point(_Point):
    # TODO: think about what's fundamental in Point

    def __init__(self, id_, pv_surface=None, vtk_path=None, dirname=None):
        super().__init__(id_=id_, pv_surface=pv_surface)

        if vtk_path is None and dirname is None:
            raise ValueError("Need to define ``vtk_path`` or ``dirname``")

        if vtk_path is None:
            vtk_path = dirname / f"{self.id}.vtk"

        self._vtk_path = vtk_path

    @property
    def vtk_path(self):
        return self._vtk_path


class ShootedPoint(_Point):
    def __init__(self, base_point, tangent_vec, outputs_dir=None):
        id_ = f"{base_point.id}_shoot_{tangent_vec.id}"
        super().__init__(id_=id_)

        if outputs_dir is None:
            outputs_dir = Path.cwd()

        self.dirname = outputs_dir / id_

        self.base_point = base_point
        self.tangent_vec = tangent_vec

        self._vkt_path = None

    @property
    def vtk_path(self):
        if self._vkt_path is None:
            self._vkt_path = pdefoio.load_shooted_point(self.dirname, as_path=True)

        return self._vkt_path

    def flow(self, as_path=False):
        # TODO: this is fundamentally a geodesic (but here we do not have control_points)
        # TODO: this should go in tangent vec only?
        return pdefoio.load_shooting_flow(
            self.dirname,
            as_pv=True,
            as_path=as_path,
        )


class ReconstructedPoint(_Point):
    def __init__(self, tangent_vec):
        # NB: point_id is required for DeterministicAtlas case

        # TODO: think about id
        super().__init__(id_=f"{tangent_vec.id}_r")

        self.tangent_vec = tangent_vec
        self._vkt_path = None

    @property
    def base_point(self):
        return self.tangent_vec.base_point

    @property
    def point(self):
        return self.tangent_vec.point

    @property
    def dirname(self):
        return self.tangent_vec.dirname

    @property
    def flow(self, as_path=False):
        # TODO: call it associated Geodesic?
        return self.tangent_vec.flow(as_path=as_path)

    @property
    def vtk_path(self):
        if self._vkt_path is None:
            self._vkt_path = pdefoio.load_deterministic_atlas_reconstruction(
                self.dirname, as_path=True, id_=self.point.id
            )

        return self._vkt_path


class DeterministicAtlasPoint(_Point):
    def __init__(self, id_, points, outputs_dir=None):
        super().__init__(id_=id_)

        if outputs_dir is None:
            outputs_dir = Path.cwd()

        self.dirname = outputs_dir / id_
        self.points = points

        # TODO: flows (as dict?)
        # TODO: actually load them as a registration?

        self._vkt_path = None

    def control_points(self, as_path=True):
        return pdefoio.load_cp(self.dirname, as_path=as_path)

    @property
    def vtk_path(self):
        return pdefoio.load_template(self.dirname, as_path=True)

    def tangent_vecs(self):
        # TODO: allow specification of id?
        return [TangentVecFromAtlas(self, point) for point in self.points]


class TangentVecFromRegistration:
    def __init__(self, base_point, point, outputs_dir=None):
        self.id = f"{base_point.id}_to_{point.id}"

        if outputs_dir is None:
            outputs_dir = Path.cwd()

        self.dirname = outputs_dir / self.id

        self.base_point = base_point
        self.point = point

    def control_points(self, as_path=True):
        return pdefoio.load_cp(self.dirname, as_path=as_path)

    def momenta(self, as_path=True):
        return pdefoio.load_momenta(self.dirname, as_path=as_path)

    def reconstructed(self, as_path=False):
        return ReconstructedPoint(tangent_vec=self)

    def flow(self, as_path=False):
        # TODO: geodesic, last
        return pdefoio.load_deterministic_atlas_flow(
            self.dirname, as_pv=True, as_path=as_path
        )


class TangentVecFromAtlas:
    # TODO: create notion of TangentVector
    # TODO: this is almost the same as TangentVecFromRegistration, merge?
    def __init__(self, atlas, point):
        self.id = f"{atlas.id}_to_{point.id}"

        self.base_point = atlas
        self.point = point

    @property
    def dirname(self):
        return self.base_point.dirname

    def reconstructed(self, as_path=False):
        return ReconstructedPoint(tangent_vec=self)

    def control_points(self, as_path=True):
        return pdefoio.load_cp(self.dirname, as_path=as_path)

    def momenta(self, as_path=True):
        return pdefoio.load_deterministic_atlas_momenta(
            self.dirname, as_path=as_path, id_=self.point.id
        )

    def flow(self, as_path=False):
        return pdefoio.load_deterministic_atlas_flow(
            self.dirname, as_pv=True, as_path=as_path, id_=self.point.id
        )


class _TransportedVec:
    # TODO: think about need
    # TODO: this is the pole_ladder thing
    def __init__(self, vec, base_point, direction, outputs_dir=None, pole_ladder=True):
        scheme = "ladder" if pole_ladder else "fan"
        self.id = f"{vec.id}_along_{scheme}_{direction.id}"
        if outputs_dir is None:
            outputs_dir = Path.cwd()

        self.dirname = outputs_dir / self.id

        self.vec = vec
        self.base_point = base_point
        self.direction = direction

    def control_points(self, as_path=True):
        return pdefoio.load_transported_cp(self.dirname, as_path=as_path)

    def momenta(self, as_path=True):
        return pdefoio.load_transported_momenta(self.dirname, as_path=as_path)


class TransportedVectorFan(_TransportedVec):
    # TODO: parallel
    # TODO: can load more stuff
    # TODO: add shooted_reconstructed

    # NB: they choose to use the same control points for geodesic and pt
    def __init__(self, tangent_vec, base_point, direction, outputs_dir=None):
        super().__init__(
            tangent_vec,
            base_point,
            direction,
            outputs_dir=outputs_dir,
            pole_ladder=False,
        )

    def reconstructed(self):
        # NB: it reconstructs the end point of direction

        # TODO: control dirname? or maybe only point?
        # TODO: add flow? (not relevant right now)
        id_ = f"{self.direction.id}_r"
        vtk_path = pdefoio.load_shooted_point(self.dirname, as_path=True)
        return Point(id_=id_, vtk_path=vtk_path)

    def reconstructed_shooted(self):
        id_ = f"{self.direction.id}_rs"
        vtk_path = pdefoio.load_parallel_shooted_point(self.dirname, as_path=True)
        return Point(id_=id_, vtk_path=vtk_path)


def TransportedVec(*args, pole_ladder=True, **kwargs):
    if pole_ladder:
        return _TransportedVec(*args, pole_ladder=True, **kwargs)

    return TransportedVectorFan(*args, **kwargs)


__all__ = auto_all(globals())

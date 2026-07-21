import abc

import geomstats.backend as gs
from geomstats.geometry.discrete_surfaces import (  # noqa: F401
    DiscreteSurfaces,
)
from geomstats.geometry.discrete_surfaces import (
    L2SurfacesMetric as GsL2SurfacesMetric,
)
from geomstats.geometry.stratified.point_set import PointSet as _PointSet
from geomstats.geometry.stratified.vectorization import (
    _manipulate_output as gs_manipulate_output,
)
from geomstats.geometry.stratified.vectorization import (
    vectorize_point,
)

from polpo.mesh.surface import PvSurface
from polpo.preprocessing.mesh.conversion import PvFromData

try:
    from .fm import FmAlignerAlgorithm, vertices_to_geomfum  # noqa: F401
except ImportError:
    pass

# TODO: need homogenizaton with geomstats


class _NotImplementedMixins:
    # TODO: make this macro-based?
    def belongs(self, *args, **kwargs):
        raise NotImplementedError

    def random_point(self, *args, **kwargs):
        raise NotImplementedError


class PointSet(_NotImplementedMixins, _PointSet):
    def equip_with_metric(self, Metric=None, **metric_kwargs):
        super().equip_with_metric(Metric=Metric, **metric_kwargs)
        return self


class SurfacesSpace(PointSet):
    def __init__(self):
        super().__init__(equip=False)


class Metric(abc.ABC):
    # TODO: do a better job in geomstats?
    def __init__(self, space):
        self._space = space

    def dist(self, point_a, point_b):
        return gs.sqrt(self.squared_dist(point_a, point_b))

    @abc.abstractmethod
    def squared_dist(self, point_a, point_b):
        pass


class PullbackMetric(Metric):
    def __init__(self, space, forward_map, image_space):
        super().__init__(space)

        self.image_space = image_space
        self.forward_map = forward_map

    def squared_dist(self, point_a, point_b):
        image_point_a = self.forward_map(point_a)
        image_point_b = self.forward_map(point_b)

        return self.image_space.metric.squared_dist(image_point_a, image_point_b)


class L2SurfacesMetric(GsL2SurfacesMetric):
    # TODO: implement in geomstats

    def parallel_transport(
        self, tangent_vec, base_point=None, direction=None, end_point=None
    ):
        r"""Compute the parallel transport of a tangent vector.

        On a Euclidean space, the parallel transport of a (tangent) vector
        returns the vector itself.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point to be transported.
        base_point : array-like, shape=[..., dim]
            Point on the manifold. Point to transport from.
            Optional, default: None
        direction : array-like, shape=[..., dim]
            Tangent vector at base point, along which the parallel transport
            is computed.
            Optional, default: None.
        end_point : array-like, shape=[..., dim]
            Point on the manifold. Point to transport to.
            Optional, default: None.

        Returns
        -------
        transported_tangent_vec : array-like, shape=[..., dim]
            Transported tangent vector at `exp_(base_point)(tangent_vec_b)`.
        """
        # TODO: fix vectorization
        return gs.copy(tangent_vec)


def _output_as_array(out, to_list):
    return gs_manipulate_output(out, to_list, manipulate_output_iterable=gs.array)


@vectorize_point((0, "point"), manipulate_output=_output_as_array)
def mesh_to_array(point):
    return [point_.vertices for point_ in point]


class ArrayToPvSurface:
    def __init__(self, faces):
        self.faces = faces
        self._from_data = PvFromData() + PvSurface

    def __call__(self, point):
        if len(point.shape) == 2:
            return self._from_data((point, self.faces))

        return [self._from_data((point_, self.faces)) for point_ in point]


class PvSurfaceToArray:
    def __init__(self, faces):
        self._array_to_pv_surface = ArrayToPvSurface(faces)

    def __call__(self, base_point):
        return mesh_to_array(base_point)

    def inverse(self, image_point):
        return self._array_to_pv_surface(image_point)

    def tangent(self, tangent_vec, base_point=None, image_point=None):
        # TODO: need to check vectorization
        return gs.asarray(tangent_vec)

    def inverse_tangent(self, image_tangent_vec, image_point=None, base_point=None):
        # TODO: need to check vectorization
        return gs.asarray(image_tangent_vec)

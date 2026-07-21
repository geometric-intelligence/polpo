import abc

import geomstats.backend as gs
from geomstats.geometry.discrete_surfaces import (  # noqa: F401
    DiscreteSurfaces,
)
from geomstats.metric_geometry.point_set import PointSet as _PointSet

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

import geomstats.backend as gs
from geomstats.metric_geometry.vectorization import (
    _manipulate_output as gs_manipulate_output,
)
from geomstats.metric_geometry.vectorization import (
    vectorize_point,
)

from polpo.preprocessing.mesh.conversion import PvFromData
from polpo.surface_mesh.core import PvSurface


def _output_as_array(out, to_list):
    return gs_manipulate_output(out, to_list, manipulate_output_iterable=gs.array)


@vectorize_point((0, "point"), manipulate_output=_output_as_array)
def mesh_to_vertices(point):
    return [point_.vertices for point_ in point]


class VerticesToPvSurface:
    def __init__(self, faces):
        self.faces = faces
        self._from_data = PvFromData() + PvSurface

    def __call__(self, point):
        if len(point.shape) == 2:
            return self._from_data((point, self.faces))

        return [self._from_data((point_, self.faces)) for point_ in point]


class PvSurfaceToVertices:
    def __init__(self, faces):
        self._array_to_pv_surface = VerticesToPvSurface(faces)

    def __call__(self, base_point):
        return mesh_to_vertices(base_point)

    def inverse(self, image_point):
        return self._array_to_pv_surface(image_point)

    def tangent(self, tangent_vec, base_point=None, image_point=None):
        # TODO: need to check vectorization
        return gs.asarray(tangent_vec)

    def inverse_tangent(self, image_tangent_vec, image_point=None, base_point=None):
        # TODO: need to check vectorization
        return gs.asarray(image_tangent_vec)

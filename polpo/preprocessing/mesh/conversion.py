import copy
import sys

try:
    from ._trimesh import DataFromTrimesh, TrimeshFromData, TrimeshFromPv  # noqa:F401
except ImportError:
    pass

try:
    from ._pyvista import PvFromData, PvFromTrimesh  # noqa:F401
except ImportError:
    pass

try:
    from ._skshapes import PvFromSks, SksFromPv  # noqa:F401
except ImportError:
    pass

try:
    from ._geomstats import SurfaceFromTrimesh, TrimeshSurfaceFromTrimesh  # noqa:F401
except ImportError:
    pass

from polpo.macro import create_to_classes_from_from
from polpo.preprocessing.base import PreprocessingStep


class ToVertices(PreprocessingStep):
    def apply(self, mesh):
        return mesh.vertices


class ToFaces(PreprocessingStep):
    def apply(self, mesh):
        return mesh.faces


class FromCombinatorialStructure(PreprocessingStep):
    def __init__(self, mesh=None):
        self.mesh = mesh

    def apply(self, data):
        if self.mesh is None:
            mesh, vertices = data
        else:
            mesh = self.mesh
            vertices = data

        mesh = copy.copy(mesh)
        mesh.vertices = vertices

        return mesh


create_to_classes_from_from(sys.modules[__name__])

import copy

try:
    from ._trimesh import TrimeshFromData, TrimeshToData  # noqa:F401
except ImportError:
    pass

try:
    from ._pyvista import PvFromTrimesh  # noqa:F401
except ImportError:
    pass

try:
    from ._skshapes import SksFromPv, SksToPv  # noqa:F401
except ImportError:
    pass

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

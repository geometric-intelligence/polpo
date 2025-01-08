import numpy as np

from polpo.preprocessing.base import PreprocessingStep


class MeshCenterer(PreprocessingStep):
    def __init__(self, attr="vertices"):
        super().__init__()
        self.attr = attr

    def apply(self, mesh):
        """Center a mesh by putting its barycenter at origin of the coordinates.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            Mesh to center.

        Returns
        -------
        centered_mesh : trimesh.Trimesh
            Centered Mesh.
        hippocampus_center: coordinates of center of the mesh before centering
        """
        values = getattr(mesh, self.attr)
        center = np.mean(values, axis=0)
        setattr(mesh, self.attr, values - center)

        return mesh


class MeshScaler(PreprocessingStep):
    def __init__(self, scaling_factor=20.0, attr="vertices"):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.attr = attr

    def apply(self, mesh):
        values = getattr(mesh, self.attr)
        setattr(mesh, self.attr, values / self.scaling_factor)
        return mesh


class TransformVertices(PreprocessingStep):
    def apply(self, data):
        # TODO: accept transformation at init?
        # TODO: consider in place?

        mesh, transformation = data

        rotation_matrix = transformation[:3, :3]
        translation = transformation[:3, 3]

        mesh.vertices = (rotation_matrix @ mesh.vertices.T).T + translation

        return mesh

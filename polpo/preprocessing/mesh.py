import copy
import os

import fast_simplification

try:
    import H2_SurfaceMatch.H2_match  # noqa: E402
    import H2_SurfaceMatch.utils.utils  # noqa: E402
except ImportError:
    pass
import numpy as np
import trimesh
from sklearn.neighbors import NearestNeighbors

from .base import PreprocessingStep


class TrimeshFromData(PreprocessingStep):
    def apply(self, mesh):
        # TODO: make a check for colors?
        vertices, faces, colors = mesh
        return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)


class TrimeshToData(PreprocessingStep):
    def apply(self, mesh):
        return (
            np.array(mesh.vertices),
            np.array(mesh.faces),
            np.array(mesh.visual.vertex_colors),
        )


class MeshCenterer(PreprocessingStep):
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
        vertices = mesh.vertices
        center = np.mean(vertices, axis=0)
        mesh.vertices = vertices - center

        return mesh


class MeshScaler(PreprocessingStep):
    def __init__(self, scaling_factor=20.0):
        super().__init__()
        self.scaling_factor = scaling_factor

    def apply(self, mesh):
        mesh.vertices = mesh.vertices / self.scaling_factor
        return mesh


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


class TrimeshFaceRemoverByArea(PreprocessingStep):
    # TODO: generalize?

    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def apply(self, mesh):
        face_mask = ~np.less(mesh.area_faces, self.threshold)
        mesh.update_faces(face_mask)

        return mesh


class TrimeshDegenerateFacesRemover(PreprocessingStep):
    """Trimesh degenerate faces remover.

    Parameters
    ----------
    height: float
        Identifies faces with an oriented bounding box shorter than
        this on one side.
    """

    def __init__(self, height=1e-08):
        self.height = height

    def apply(self, mesh):
        mesh.update_faces(mesh.nondegenerate_faces(height=self.height))
        return mesh


class TrimeshDecimator(PreprocessingStep):
    """Trimesh simplify quadratic decimation.

    Parameters
    ----------
    percent : float
        A number between 0.0 and 1.0 for how much.
    face_count : int
        Target number of faces desired in the resulting mesh.
    agression: int
        An integer between 0 and 10, the scale being roughly 0 is
        “slow and good” and 10 being “fast and bad.”
    """

    # NB: uses fast-simplification

    def __init__(self, percent=None, face_count=None, agression=None):
        super().__init__()
        self.percent = percent
        self.face_count = face_count
        self.agression = agression

    def apply(self, mesh):
        # TODO: issue with colors?
        decimated_mesh = mesh.simplify_quadric_decimation(
            percent=self.percent,
            face_count=self.face_count,
            aggression=self.agression,
        )

        # # TODO: delete
        # colors_ = np.array(decimated_mesh.visual.vertex_colors)
        # print("unique colors after decimation:", len(np.unique(colors_, axis=0)))

        return decimated_mesh


class FastSimplificationDecimator(PreprocessingStep):
    def __init__(self, target_reduction=0.25):
        super().__init__()
        self.target_reduction = target_reduction
        self._nbrs = NearestNeighbors(n_neighbors=1)

    def apply(self, mesh):
        # TODO: make a check for colors?
        vertices, faces, colors = mesh

        vertices_, faces_ = fast_simplification.simplify(
            vertices, faces, self.target_reduction
        )

        # TODO: can this be done better?
        self._nbrs.fit(vertices)
        _, indices = self._nbrs.kneighbors(vertices_)
        indices = np.squeeze(indices)
        colors_ = colors[indices]

        return vertices_, faces_, colors_


class H2MeshDecimator(PreprocessingStep):
    def __init__(self, decimation_factor=10.0):
        super().__init__()
        self.decimation_factor = decimation_factor

    def apply(self, mesh):
        # TODO: issues using due to open3d (delete?)

        # TODO: make a check for colors?
        vertices, faces, colors = mesh

        n_faces_after_decimation = faces.shape[0] // self.decimation_factor
        (
            vertices_after_decimation,
            faces_after_decimation,
            colors_after_decimation,
        ) = H2_SurfaceMatch.utils.utils.decimate_mesh(
            vertices, faces, n_faces_after_decimation, colors=colors
        )

        return [
            vertices_after_decimation,
            faces_after_decimation,
            colors_after_decimation,
        ]


class TrimeshToPly(PreprocessingStep):
    def __init__(self, dirname=""):
        self.dirname = dirname
        # TODO: create dir if does not exist?

        # TODO: add override?

    def apply(self, data):
        filename, mesh = data

        ext = ".ply"
        if not filename.endswith(ext):
            filename += ext

        path = os.path.join(self.dirname, filename)

        ply_text = trimesh.exchange.ply.export_ply(
            mesh, encoding="binary", include_attributes=True
        )

        # TODO: add verbose
        # print(f"- Write mesh to {filename}")
        with open(path, "wb") as file:
            file.write(ply_text)

        return data


class TrimeshReader(PreprocessingStep):
    # TODO: update
    def apply(self, path):
        return trimesh.load(path)


# TODO: create mesh serializer


class H2MeshAligner(PreprocessingStep):
    def __init__(
        self,
        a0=0.01,
        a1=10.0,
        b1=10.0,
        c1=1.0,
        d1=0.0,
        a2=1.0,
        resolutions=0,
        paramlist=(),
    ):
        super().__init__()
        self.a0 = a0
        self.a1 = a1
        self.b1 = b1
        self.c1 = c1
        self.d1 = d1
        self.a2 = a2
        self.resolutions = resolutions
        self.paramlist = paramlist

        # TODO: allow control of device
        self.device = None

    def apply(self, meshes):
        target_mesh, template_mesh = meshes

        # TODO: upgrade device?
        geod, F0, color0 = H2_SurfaceMatch.H2_match.H2MultiRes(
            source=template_mesh,
            target=target_mesh,
            a0=self.a0,
            a1=self.a1,
            b1=self.b1,
            c1=self.c1,
            d1=self.d1,
            a2=self.a2,
            resolutions=self.resolutions,
            start=None,
            paramlist=self.paramlist,
            device=self.device,
        )

        return geod[-1], F0, color0


class IdentityMeshAligner(PreprocessingStep):
    # useful for debugging

    def apply(self, meshes):
        target_mesh, _ = meshes
        return target_mesh

import os

import numpy as np
import trimesh

from polpo.preprocessing.base import PreprocessingStep


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

import numpy as np
import pyvista as pv

from polpo.preprocessing.base import PreprocessingStep


class PvFromData(PreprocessingStep):
    """Convert arrays into pv.PolyData.

    Parameters
    ----------
    keep_colors : bool
        Whether to keep colors if present.
    """

    def __init__(self, keep_colors=True):
        super().__init__()
        self.keep_colors = keep_colors

    def __call__(self, mesh):
        """Apply step.

        Parameters
        ----------
        mesh : tuple
            Mesh represented by (vertices, faces, colors) or (vertices, faces).

        Returns
        -------
        mesh : pv.PolyData
            Converted mesh.
        """
        if len(mesh) == 3:
            vertices, faces, colors = mesh
        else:
            vertices, faces = mesh
            colors = None

        poly_data = pv.PolyData.from_regular_faces(points=vertices, faces=faces)
        if self.keep_colors and colors is not None:
            poly_data["colors"] = colors
        return poly_data


class DataFromPv(PreprocessingStep):
    """Convert pv.PolyData into arrays.

    Parameters
    ----------
    keep_colors : bool
        Whether to keep colors if present.
    """

    def __init__(self, keep_colors=True):
        super().__init__()
        self.keep_colors = keep_colors

    def __call__(self, mesh):
        vertices, faces = (
            np.array(mesh.points),
            np.array(mesh.regular_faces),
        )
        if self.keep_colors and "colors" in mesh.array_names:
            return vertices, faces, np.array(mesh["colors"])

        return vertices, faces

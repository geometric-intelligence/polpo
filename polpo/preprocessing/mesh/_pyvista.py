import os

import numpy as np
import pyvista as pv
from sklearn.neighbors import NearestNeighbors

from polpo.preprocessing.base import PreprocessingStep
from polpo.utils import params_to_kwargs


class PvAlign(PreprocessingStep):
    """Align a dataset to another.

    https://docs.pyvista.org/api/core/_autosummary/pyvista.datasetfilters.align
    """

    def __init__(
        self,
        max_landmarks=100,
        max_mean_distance=1e-05,
        max_iterations=500,
        check_mean_distance=True,
        start_by_matching_centroids=True,
    ):
        self.max_landmarks = max_landmarks
        self.max_mean_distance = max_mean_distance
        self.max_iterations = max_iterations
        self.check_mean_distance = check_mean_distance
        self.start_by_matching_centroids = start_by_matching_centroids

    def apply(self, data):
        source, target = data
        return source.align(
            target,
            **params_to_kwargs(self),
        )


class PvSmoothTaubin(PreprocessingStep):
    """Smooth a PolyData DataSet with Taubin smoothing.

    https://docs.pyvista.org/api/core/_autosummary/pyvista.polydatafilters.smooth_taubin#pyvista.PolyDataFilters.smooth_taubin"
    """

    def __init__(
        self,
        n_iter=20,
        pass_band=0.1,
        edge_angle=15.0,
        feature_angle=45.0,
        boundary_smoothing=True,
        feature_smoothing=False,
        non_manifold_smoothing=False,
        normalize_coordinates=False,
        inplace=False,
        progress_bar=False,
    ):
        self.n_iter = n_iter
        self.pass_band = pass_band
        self.edge_angle = edge_angle
        self.feature_angle = feature_angle
        self.boundary_smoothing = boundary_smoothing
        self.feature_smoothing = feature_smoothing
        self.non_manifold_smoothing = non_manifold_smoothing
        self.normalize_coordinates = normalize_coordinates
        self.inplace = inplace
        self.progress_bar = progress_bar

    def apply(self, poly_data):
        return poly_data.smooth_taubin(**params_to_kwargs(self))


class PvDecimate(PreprocessingStep):
    """Reduce the number of triangles in a triangular mesh.

    Uses vtkQuadricDecimation.

    https://docs.pyvista.org/api/core/_autosummary/pyvista.polydatafilters.decimate#pyvista.PolyDataFilters.decimate
    """

    def __init__(
        self,
        target_reduction,
        volume_preservation=False,
        attribute_error=False,
        scalars=True,
        vectors=True,
        normals=False,
        tcoords=True,
        tensors=True,
        scalars_weight=0.1,
        vectors_weight=0.1,
        normals_weight=0.1,
        tcoords_weight=0.1,
        tensors_weight=0.1,
        inplace=False,
        progress_bar=False,
        keep_colors=True,
    ):
        self.target_reduction = target_reduction
        self.volume_preservation = volume_preservation
        self.attribute_error = attribute_error
        self.scalars = scalars
        self.vectors = vectors
        self.normals = normals
        self.tcoords = tcoords
        self.tensors = tensors
        self.scalars_weight = scalars_weight
        self.vectors_weight = vectors_weight
        self.normals_weight = normals_weight
        self.tcoords_weight = tcoords_weight
        self.tensors_weight = tensors_weight
        self.inplace = inplace
        self.progress_bar = progress_bar

        self.keep_colors = keep_colors
        self._nbrs = NearestNeighbors(n_neighbors=1)

    def apply(self, poly_data):
        colors = None
        if self.keep_colors and "colors" in poly_data.array_names:
            colors = poly_data["colors"]
            vertices = poly_data.points

        poly_data = poly_data.decimate(
            **params_to_kwargs(self, ignore_private=True, ignore=("keep_colors",))
        )

        if colors is None:
            return poly_data

        self._nbrs.fit(vertices)
        _, indices = self._nbrs.kneighbors(poly_data.points)
        indices = np.squeeze(indices)
        poly_data["colors"] = colors[indices]

        return poly_data


class PvFromTrimesh(PreprocessingStep):
    def __init__(self, store_colors=True):
        self.store_colors = store_colors

    def apply(self, mesh):
        pvmesh = pv.wrap(mesh)
        if self.store_colors and hasattr(mesh.visual, "vertex_colors"):
            pvmesh["colors"] = np.array(mesh.visual.vertex_colors)

        return pvmesh


class PvWriter(PreprocessingStep):
    """
    https://docs.pyvista.org/api/core/_autosummary/pyvista.polydata.save
    """

    def __init__(
        self,
        dirname="",
        ext=None,
        binary=True,
        recompute_normals=False,
    ):
        self.dirname = dirname
        self.ext = ext

        self.binary = binary
        self.recompute_normals = recompute_normals

    def apply(self, data):
        # TODO: create dir if does not exist?
        # filename extension ignored if ext is not None
        filename, poly_data = data

        if self.ext is not None:
            ext = self.ext
            if "." in filename:
                filename = filename.split(".")[0]
            filename += f".{self.ext}"
        else:
            ext = filename.split(".")[1]

        path = os.path.join(self.dirname, filename)

        texture = (
            poly_data["colors"]
            if ext == "ply" and "colors" in poly_data.array_names
            else None
        )

        poly_data.save(
            path,
            binary=self.binary,
            texture=texture,
            recompute_normals=self.recompute_normals,
        )

        return path


class PvReader(PreprocessingStep):
    def __init__(self, rename_colors=True):
        self.rename_colors = rename_colors

    def apply(self, filename):
        poly_data = pv.read(filename)

        if "RGBA" in poly_data.array_names:
            poly_data.rename_array("RGBA", "colors")
        elif "RGB" in poly_data.array_names:
            poly_data.rename_array("RGB", "colors")

        return poly_data


class PvExtractLargest(PreprocessingStep):
    """Extract largest connected set in mesh.

    https://docs.pyvista.org/api/core/_autosummary/pyvista.datasetfilters.extract_largest#pyvista.DataSetFilters.extract_largest
    """

    def __init__(self, inplace=False, progress_bar=False):
        self.inplace = inplace
        self.progress_bar = progress_bar

    def apply(self, poly_data):
        return poly_data.extract_largest(**params_to_kwargs(self))

import os
from pathlib import Path

import numpy as np
import pyvista as pv
from sklearn.neighbors import NearestNeighbors

from polpo.preprocessing.base import PreprocessingStep, RegistrationStep
from polpo.preprocessing.mesh._register import register_vertices_attr
from polpo.utils import params_to_kwargs

register_vertices_attr(pv.PolyData, "points")


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


class PvAlign(RegistrationStep):
    """Align a dataset to another with ICP.

    https://docs.pyvista.org/api/core/_autosummary/pyvista.datasetfilters.align
    """

    def __init__(
        self,
        target=None,
        max_landmarks=100,
        max_mean_distance=1e-05,
        max_iterations=500,
        check_mean_distance=True,
        start_by_matching_centroids=True,
        return_matrix=False,
    ):
        super().__init__(target=target)
        self.max_landmarks = max_landmarks
        self.max_mean_distance = max_mean_distance
        self.max_iterations = max_iterations
        self.check_mean_distance = check_mean_distance
        self.start_by_matching_centroids = start_by_matching_centroids
        self.return_matrix = return_matrix

    def __call__(self, data):
        """Apply step.

        Parameters
        ----------
        data : pv.Polydata or tuple[pv.PolyData; 2]
            (source, target) meshes.

        Returns
        -------
        mesh : pv.PolyData
            Source aligned to target.
        matrix : numpy.ndarray
            Transform matrix to transform the input dataset to the target dataset.
        """
        source, target = self._get_source_and_target(data)

        return source.align(
            target,
            **params_to_kwargs(self, ignore=("target",)),
        )


class PvSmoothTaubin(PreprocessingStep):
    """Smooth a PolyData DataSet with Taubin smoothing.

    https://docs.pyvista.org/api/core/_autosummary/pyvista.polydatafilters.smooth_taubin#pyvista.PolyDataFilters.smooth_taubin
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

    def __call__(self, poly_data):
        """Apply step."""
        return poly_data.smooth_taubin(**params_to_kwargs(self))


class PvDecimate(PreprocessingStep):
    """Reduce the number of triangles in a triangular mesh.

    Uses vtkQuadricDecimation.

    https://docs.pyvista.org/api/core/_autosummary/pyvista.polydatafilters.decimate#pyvista.PolyDataFilters.decimate
    """

    def __init__(
        self,
        target_reduction,
        volume_preservation=True,
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
        super().__init__()
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

    def __call__(self, poly_data):
        """Apply step.

        Parameters
        ----------
        mesh : pv.PolyData
            Mesh to decimate.

        Returns
        -------
        decimated_mesh : pv.PolyData
            Decimated mesh.
        """
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
    """Convert trimesh.Trimesh into pv.PolyData."""

    def __init__(self, store_colors=True):
        super().__init__()
        self.store_colors = store_colors

    def __call__(self, mesh):
        """Apply step.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            Mesh to convert.

        Returns
        -------
        mesh : pv.PolyData
            Converted mesh.
        """
        pvmesh = pv.wrap(mesh)
        if self.store_colors and hasattr(mesh.visual, "vertex_colors"):
            pvmesh["colors"] = np.array(mesh.visual.vertex_colors)

        return pvmesh


class PvWriter(PreprocessingStep):
    """Write a surface mesh to disk.

    https://docs.pyvista.org/api/core/_autosummary/pyvista.polydata.save
    """

    def __init__(
        self,
        dirname="",
        ext=None,
        binary=True,
        recompute_normals=False,
        exists_ok=True,
    ):
        super().__init__()
        self.dirname = dirname
        self.ext = ext
        self.exists_ok = exists_ok

        self.binary = binary
        self.recompute_normals = recompute_normals

    def __call__(self, data):
        """Apply step.

        Parameters
        ----------
        data : tuple[str, pv.PolyData]
            Filename and mesh to write.

        Returns
        -------
        path : str
            Filename.
        """
        # TODO: create dir if does not exist?
        filename, poly_data = data

        if "." not in filename and self.ext is not None:
            filename += f".{self.ext}"

        ext = filename.split(".")[1]

        path = Path(os.path.join(self.dirname, filename))
        path.parent.mkdir(parents=True, exist_ok=self.exists_ok)

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
    """Read file.

    https://docs.pyvista.org/api/utilities/_autosummary/pyvista.read
    """

    def __init__(self, rename_colors=True):
        super().__init__()
        self.rename_colors = rename_colors

    def __call__(self, filename):
        """Apply step.

        Parameters
        ----------
        filename : str
            File name.

        Returns
        -------
        mesh : pv.PolyData
            Loaded mesh.
        """
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
        super().__init__()
        self.inplace = inplace
        self.progress_bar = progress_bar

    def __call__(self, poly_data):
        """Apply step.

        Parameters
        ----------
        poly_data : pv.PolyData
            Mesh.

        Returns
        -------
        mesh : pv.PolyData
            Largest mesh component.
        """
        return poly_data.extract_largest(**params_to_kwargs(self))


class PvExtractPoints(PreprocessingStep):
    """Get a subset of the grid (with cells).

    Cells are added if contain any of the given point indices.

    https://docs.pyvista.org/api/core/_autosummary/pyvista.datasetfilters.extract_points
    """

    def __init__(
        self,
        adjacent_cells=True,
        include_cells=True,
        progress_bar=False,
        extract_surface=True,
    ):
        super().__init__()
        self.adjacent_cells = adjacent_cells
        self.include_cells = include_cells
        self.progress_bar = progress_bar
        self.extract_surface = extract_surface

    def __call__(self, data):
        """Apply step.

        Parameters
        ----------
        data : tuple[pv.PolyData, indices]
        """
        mesh, indices = data
        subset = mesh.extract_points(
            indices, **params_to_kwargs(self, func=mesh.extract_points)
        )
        if self.extract_surface:
            return subset.extract_surface()

        return subset


class PvSelectColor(PreprocessingStep):
    """Get subset of a mesh with given color."""

    def __init__(
        self,
        color=None,
        adjacent_cells=True,
        include_cells=True,
        progress_bar=False,
        extract_surface=True,
    ):
        super().__init__()
        self.color = color
        self._point_extractor = PvExtractPoints(
            adjacent_cells=adjacent_cells,
            include_cells=include_cells,
            progress_bar=progress_bar,
            extract_surface=extract_surface,
        )

    def __call__(self, data):
        if isinstance(data, (list, tuple)):
            mesh, color = data
        else:
            mesh = data
            color = self.color

        (ind,) = np.where(np.all(np.equal(mesh.get_array("colors"), color), axis=1))

        mesh = self._point_extractor((mesh, ind))

        if "colors" in mesh.array_names:
            mesh.point_data.remove("colors")

        return mesh

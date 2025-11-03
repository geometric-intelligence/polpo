import geomstats.backend as gs
import numpy as np
from geomstats.geometry.discrete_surfaces import Surface as GsSurface

# TODO: add enclosing ball algorithm and wrap geomstats surface?

# TODO: add tests checking results are the same for all representations


class Surface(GsSurface):
    # TODO: add miniball to compute bounding sphere?
    @property
    def bounds(self):
        return gs.stack(
            (gs.amin(self.vertices, axis=0), gs.amax(self.vertices, axis=0))
        )

    @property
    def vertex_centroid(self):
        return gs.mean(self.vertices, axis=0)


class _MeshDispatchMixins:
    def __init__(self, mesh):
        self._mesh = mesh

    def __getattr__(self, name):
        """Get attribute.

        It is only called when ``__getattribute__`` fails.
        Delegates attribute calling to _trimesh.
        """
        # TODO: may need to check dtype for faces
        out = getattr(self._mesh, name)
        # TODO: may need to improve logic
        if isinstance(out, np.ndarray):
            return gs.asarray(out)

        return out


class TrimeshSurface(_MeshDispatchMixins):
    # polymorphic to `Surface` and `Trimesh`
    def __init__(self, trimesh, signal=None):
        super().__init__(trimesh)
        self.signal = signal

    @property
    def face_centroids(self):
        return gs.from_numpy(self._mesh.triangles_center)

    @property
    def face_areas(self):
        return gs.expand_dims(gs.from_numpy(self._mesh.area_faces), axis=1)

    @property
    def vertex_centroid(self):
        return gs.from_numpy(self._mesh.centroid)


class PvSurface(_MeshDispatchMixins):
    # https://docs.pyvista.org/api/core/_autosummary/pyvista.polydata
    # polymorphic to `Surface` and `pyvista.PolyData`

    def __init__(self, pv_mesh, signal=None):
        super().__init__(pv_mesh)
        self.signal = signal

    @property
    def vertices(self):
        return gs.asarray(self._mesh.points)

    @property
    def faces(self):
        return gs.asarray(self._mesh.regular_faces)

    @property
    def face_areas(self):
        return gs.expand_dims(
            gs.asarray(self._mesh.compute_cell_sizes()["Area"]), axis=1
        )

    @property
    def face_centroids(self):
        return gs.asarray(
            self._mesh.cell_centers().points, dtype=gs.get_default_dtype()
        )

    @property
    def face_normals(self):
        return gs.asarray(self._mesh.face_normals, dtype=gs.get_default_dtype())

    @property
    def bounds(self):
        return gs.moveaxis(gs.reshape(gs.asarray(self._mesh.bounds), (3, 2)), 0, 1)

    @property
    def vertex_centroid(self):
        return gs.mean(self.vertices, axis=0)

    @property
    def edges(self):
        """Edges of the mesh.

        Returns
        -------
        edges : array-like, shape=[n_edges, 2]
        """
        vind012 = gs.concatenate([self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]])
        vind120 = gs.concatenate([self.faces[:, 1], self.faces[:, 2], self.faces[:, 0]])
        edges = gs.stack(
            [
                gs.concatenate([vind012, vind120]),
                gs.concatenate([vind120, vind012]),
            ],
            axis=-1,
        )
        edges = gs.unique(edges, axis=0)
        return edges[edges[:, 1] > edges[:, 0]]

    @property
    def edge_lengths(self):
        edge_points = self.vertices[self.edges]

        return gs.linalg.norm(edge_points[..., 0, :] - edge_points[..., 1, :], axis=-1)

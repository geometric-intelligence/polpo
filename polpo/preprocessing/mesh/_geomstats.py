import geomstats.backend as gs
import numpy as np
from geomstats.varifold import Surface

from polpo.preprocessing.base import PreprocessingStep


class TrimeshSurface:
    # polymorphic to `Surface` and `Trimesh`
    def __init__(self, trimesh, signal=None):
        self._trimesh = trimesh
        self.signal = signal

    @property
    def face_centroids(self):
        return gs.from_numpy(self._trimesh.triangles_center)

    @property
    def face_areas(self):
        return gs.expand_dims(gs.from_numpy(self._trimesh.area_faces), axis=1)

    def __getattr__(self, name):
        """Get attribute.

        It is only called when ``__getattribute__`` fails.
        Delegates attribute calling to _trimesh.
        """
        # TODO: may need to check dtype for faces
        out = getattr(self._trimesh, name)
        if isinstance(out, np.ndarray):
            return gs.from_numpy(out)

        return out


class SurfaceFromTrimesh(PreprocessingStep):
    def apply(self, trimesh):
        return Surface(
            gs.from_numpy(trimesh.vertices),
            gs.from_numpy(trimesh.faces.astype(np.int64)),
        )


class TrimeshSurfaceFromTrimesh(PreprocessingStep):
    def __init__(self, clone=False):
        self.clone = clone

    def apply(self, trimesh):
        if self.clone:
            trimesh = trimesh.copy()

        return TrimeshSurface(trimesh)

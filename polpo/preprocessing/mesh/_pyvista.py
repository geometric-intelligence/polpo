import numpy as np
import pyvista as pv

from polpo.preprocessing.base import PreprocessingStep


class PvFromTrimesh(PreprocessingStep):
    def __init__(self, store_colors=True):
        self.store_colors = store_colors

    def apply(self, mesh):
        pvmesh = pv.wrap(mesh)
        if self.store_colors and hasattr(mesh.visual, "vertex_colors"):
            pvmesh["colors"] = np.array(mesh.visual.vertex_colors)

        return pvmesh

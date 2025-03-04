from polpo.preprocessing.base import PreprocessingStep

try:
    from ._pyvista import PvAlign  # noqa:F401
except ImportError:
    pass

try:
    from ._h2_surfacematch import H2MeshAligner  # noqa:F401
except ImportError:
    pass

try:
    from ._skshapes import SksRigidRegistration  # noqa:F401
except ImportError:
    pass

import numpy as np


class IdentityMeshAligner(PreprocessingStep):
    # useful for debugging

    def apply(self, meshes):
        target_mesh, _ = meshes
        return target_mesh

class HCToTemplateTransform(PreprocessingStep):
    def __init__(self, vox2WorldHC=None, vox2WorldTemplate=None):
        super().__init__()    
        self.vox2WorldHC = vox2WorldHC["affine"] #  affine Voxel -> World (Hippocampus)
        self.vox2WorldTemplate = vox2WorldTemplate["affine"] #  affine Voxel -> World (Template - Full Brain)

    def apply(self, data):
        # Inverse affine matrix for brain (World -> Voxel)
        vox2WorldTemplateInv = np.linalg.inv(self.vox2WorldTemplate)

        # Combine the two affine transformations (world space transformation)
        transformation_matrix = vox2WorldTemplateInv @ self.vox2WorldHC
        return transformation_matrix


import numpy as np
import trimesh

class ApplyTransform(PreprocessingStep):
    """
    Applies a 4x4 transformation matrix to a Trimesh object.
    """

    def __init__(self, mesh=None, transform=np.eye(4)):
        super().__init__()
        self.mesh = mesh
        self.transform = transform

    def apply(self, meshes):
        """
        Parameters:
        - mesh (Trimesh): The mesh to transform.
        - transform (np.ndarray): The 4x4 transformation matrix.

        Returns:
        - Trimesh: Transformed mesh.
        """
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("ApplyTransform expects the first argument to be a Trimesh object.")

        if not isinstance(transform, np.ndarray) or transform.shape != (4, 4):
            raise ValueError("ApplyTransform expects the second argument to be a 4x4 NumPy array.")

        # Convert vertices to homogeneous coordinates
        augmented_vertices = np.hstack([mesh.vertices, np.ones((mesh.vertices.shape[0], 1))])

        # Apply transformation
        transformed_vertices = (transform @ augmented_vertices.T).T[:, :3]  # Convert back to (n,3)

        # Return transformed mesh
        return trimesh.Trimesh(vertices=transformed_vertices, faces=mesh.faces, vertex_colors=mesh.visual.vertex_colors)


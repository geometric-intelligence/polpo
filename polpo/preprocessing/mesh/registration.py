from polpo.preprocessing.base import PreprocessingStep

try:
    from ._h2_surfacematch import H2MeshAligner  # noqa:F401
except ImportError:
    pass

try:
    from ._skshapes import SksRigidRegistration  # noqa:F401
except ImportError:
    pass


class IdentityMeshAligner(PreprocessingStep):
    # useful for debugging

    def apply(self, meshes):
        target_mesh, _ = meshes
        return target_mesh

import polpo.preprocessing.dict as ppdict
from polpo.preprocessing._preprocessing import Map, PartiallyInitializedStep
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


class IdentityMeshAligner(PreprocessingStep):
    # useful for debugging

    def __call__(self, meshes):
        target_mesh, _ = meshes
        return target_mesh


class RigidAlignment(PartiallyInitializedStep):
    def __init__(self, target=None, is_dict=True, **kwargs):
        # TODO: create a more robust iterator/template selector?

        if target is None:
            if is_dict:
                target = lambda x: next(iter(x.values()))
            else:
                target = lambda x: x[0]

        if is_dict:
            Step = lambda **_kwargs: ppdict.DictMap(PvAlign(**_kwargs))
        else:
            Step = lambda **_kwargs: Map(PvAlign(**_kwargs))

        super().__init__(Step=Step, _target=target, **kwargs)

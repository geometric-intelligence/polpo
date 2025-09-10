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
    def __init__(self, target=None, **kwargs):
        is_dict = lambda x: isinstance(x, dict)

        if target is None:

            def target(data):
                if is_dict(data):
                    return next(iter(data.values()))

                return data[0]

        def _Step(is_dict, **kwargs):
            if is_dict:
                return ppdict.DictMap(PvAlign(**kwargs))

            return Map(PvAlign(**kwargs))

        super().__init__(Step=_Step, _target=target, _is_dict=is_dict, **kwargs)

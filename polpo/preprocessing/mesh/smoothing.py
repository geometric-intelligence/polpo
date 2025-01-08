try:
    from ._trimesh import TrimeshLaplacianSmoothing  # noqa:F401
except ImportError:
    pass

try:
    from ._pyvista import PvSmoothTaubin  # noqa:F401
except ImportError:
    pass

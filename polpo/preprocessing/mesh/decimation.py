try:
    from ._trimesh import TrimeshDecimator  # noqa:F401
except ImportError:
    pass

try:
    from ._fast_simplification import FastSimplificationDecimator  # noqa:F401
except ImportError:
    pass

try:
    from ._pyvista import PvDecimate  # noqa:F401
except ImportError:
    pass

try:
    from ._h2_surfacematch import H2MeshDecimator  # noqa:F401
except ImportError:
    pass

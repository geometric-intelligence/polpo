try:
    from ._trimesh import (  # noqa:F401
        TrimeshDegenerateFacesRemover,
        TrimeshFaceRemoverByArea,
        TrimeshLargestComponentSelector,
    )
except ImportError:
    pass

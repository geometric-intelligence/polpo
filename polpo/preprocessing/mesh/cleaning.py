try:
    from ._trimesh import (  # noqa:F401
        TrimeshDegenerateFacesRemover,
        TrimeshFaceRemoverByArea,
    )
except ImportError:
    pass

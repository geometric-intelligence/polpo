import sys

from polpo.macro import create_to_classes_from_from

try:
    from ._open3d import O3dPointCloudFromNp  # noqa:F401
except ImportError:
    pass


create_to_classes_from_from(sys.modules[__name__])

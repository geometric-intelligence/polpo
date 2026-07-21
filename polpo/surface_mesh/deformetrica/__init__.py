import polpo.utils as _putils

HAS_DEFORMETRICA = _putils.has_package("deformetrica")

from .core import *  # noqa:F403

if HAS_DEFORMETRICA:
    # allows using repr without deformetrica
    from .geometry import LddmmMetric
    from .learning import FrechetMean

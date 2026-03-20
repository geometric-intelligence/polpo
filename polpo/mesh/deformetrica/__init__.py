import polpo.utils as _putils

HAS_DEFORMETRICA = _putils.has_package("deformetrica")

if HAS_DEFORMETRICA:
    # allows using repr without deformetrica
    from .geometry import FrechetMean, LddmmMetric


from .repr import *  # noqa:F403

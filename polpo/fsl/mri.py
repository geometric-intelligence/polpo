from polpo.preprocessing import Contains
from polpo.preprocessing.path import (
    FileFinder,
    IsFileType,
)


def SegmentationsLoader(tool="fsl_first"):
    if tool.startswith("fsl"):
        image_selector = FileFinder(
            rules=[
                IsFileType("nii.gz"),
                Contains("all_fast_firstseg"),
            ]
        )
    elif tool.startswith("fast") or tool.startswith("free"):
        image_selector = FileFinder(rules=lambda x: x == "mri") + FileFinder(
            rules=lambda x: x == "aseg.auto.mgz"
        )
    else:
        raise ValueError(f"Oops, don't know how to handle: {tool}")

    return image_selector

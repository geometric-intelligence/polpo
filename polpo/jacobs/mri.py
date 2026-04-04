import polpo.preprocessing.dict as ppdict
from polpo.preprocessing import (
    BranchingPipeline,
    CartesianProduct,
    Constant,
    IdentityStep,
    IndexSelector,
    InjectData,
    Map,
)
from polpo.preprocessing.load.fsl import (
    SegmentationsLoader as FslSegmentationsLoader,
)
from polpo.preprocessing.mri import (
    MeshExtractorFromSegmentedImage,
    MeshExtractorFromSegmentedMesh,
    MriImageLoader,
    segmtool2encoding,
)
from polpo.pyvista.conversion import PvFromData

from .path import FoldersSelector


def SegmentationsLoader(
    derivative,
    data_dir="~/.herbrain/data/maternal",
    subject_subset=None,
    session_subset=None,
    as_image=False,
):
    """Create pipeline to load segmented mri filenames.

    Parameters
    ----------
    derivative : str
        Tool used to generate derivatives.
        One of the following: "fsl*", "fast*".
    data_dir : str
        Directory where to store data.
    subset : array-like
        Subset of sessions to load. If `None`, loads all.
    subject_id : str
        Identification of the subject. If None, assumes pilot.
        One of the following: "01", "1001B", "1004B".
    as_image : bool
        Whether to load file as image.

    Returns
    -------
    pipe : Pipeline
        Pipeline to load segmented mri filenames.
    """
    # TODO: as_mesh?
    folders_selector = Constant(data_dir) + FoldersSelector(
        subject_subset=subject_subset,
        session_subset=session_subset,
        derivative=derivative,
    )

    image_selector = FslSegmentationsLoader(derivative)

    if as_image:
        image_selector += MriImageLoader()

    return folders_selector + ppdict.NestedDictMap(image_selector)


def MeshLoaderFromMri(
    derivative,
    data_dir="~/.herbrain/data/maternal",
    subject_subset=None,
    session_subset=None,
    struct_subset=None,
    split_before_meshing=False,
    n_jobs=1,
):
    # subj, session
    segmentations_loader = SegmentationsLoader(
        data_dir=data_dir,
        subject_subset=subject_subset,
        session_subset=session_subset,
        derivative=derivative,
        as_image=True,
    )

    encoding = segmtool2encoding(derivative, raise_=False)
    if struct_subset is None:
        struct_subset = encoding.structs

    if split_before_meshing:
        init_step = IdentityStep()
        to_mesh = (
            MeshExtractorFromSegmentedImage(return_colors=False, encoding=encoding)
            + PvFromData()
        )
    else:
        init_step = (
            MeshExtractorFromSegmentedImage(return_colors=True, encoding=encoding)
            + PvFromData()
        )
        to_mesh = MeshExtractorFromSegmentedMesh()

    img2mesh = ppdict.NestedDictMap(
        init_step
        + (lambda obj: [obj])
        + InjectData(struct_subset, as_first=False)
        + CartesianProduct()
        + BranchingPipeline(
            [
                Map(IndexSelector(index=1)),
                Map(
                    to_mesh,
                    n_jobs=n_jobs,
                ),
            ],
        )
        + ppdict.Hash()
    )

    pipe = segmentations_loader + img2mesh

    # subj, session, struct
    return pipe

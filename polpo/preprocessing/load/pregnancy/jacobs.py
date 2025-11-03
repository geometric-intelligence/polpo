import os
import re
from pathlib import Path

import pandas as pd

import polpo.preprocessing.dict as ppdict
import polpo.preprocessing.pd as ppd
from polpo.preprocessing import (
    BranchingPipeline,
    CartesianProduct,
    Constant,
    ExceptionToWarning,
    IdentityStep,
    IndexSelector,
    InjectData,
    Map,
)
from polpo.preprocessing.load.bids import DerivativeFoldersSelector
from polpo.preprocessing.load.fsl import (
    MeshLoader as FslMeshLoader,
)
from polpo.preprocessing.load.fsl import (
    SegmentationsLoader as FslSegmentationsLoader,
)
from polpo.preprocessing.mesh.conversion import PvFromData
from polpo.preprocessing.mri import (
    MeshExtractorFromSegmentedImage,
    MeshExtractorFromSegmentedMesh,
    MriImageLoader,
    segmtool2encoding,
)
from polpo.preprocessing.path import ExpandUser, FileFinder
from polpo.preprocessing.str import DigitFinder, StartsWith

from .pilot import TabularDataLoader as PilotTabularDataLoader

# TODO: add register ID?
MATERNAL_IDS = {"01", "1001B", "1004B", "2004B", "1009B"}


def get_subject_ids(include_pilot=True, include_male=True):
    ids = MATERNAL_IDS.copy()

    if not include_pilot:
        ids.remove("01")

    if not include_male:
        for id_ in ids.copy():
            if id_.startswith("2"):
                ids.remove(id_)

    return ids


def _session_sorter(session_id):
    # for session_id other than in pilot
    return (
        re.sub(r"\d+$", "", session_id),
        DigitFinder(index=-1)(session_id),
    )


def TabularDataLoader(
    data_dir="~/.herbrain/data/maternal", subject_subset=None, index_by_session=False
):
    """Create pipeline to load maternal csv data.

    Parameters
    ----------
    data_dir : str
        Data root dir.
    subject_subset : array-like
        Id of the subjects. If None, assumes all.
        One of the following: "01", "1001B", "1004B".
        If pilot and other, loads only common columns.
    index_by_session : bool
        Whether to index the dataframe by session.
        Only applies if one subject.

    Returns
    -------
    pipe : Pipeline
        Pipeline to load maternal csv data.
    """
    project_folder = "maternal_brain_project"
    data_dir = Path(data_dir).expanduser()

    pilot_pipe = None
    if subject_subset is None or "01" in subject_subset:
        project_folder_pilot = f"{project_folder}_pilot"

        pilot_pipe = PilotTabularDataLoader(
            data_dir=data_dir / project_folder_pilot / "rawdata",
            index_by_session=index_by_session and len(subject_subset) == 1,
        )

    if pilot_pipe and (subject_subset is not None and len(subject_subset) == 1):
        return pilot_pipe

    if subject_subset is not None:
        subject_subset = subject_subset.copy()
        subject_subset.remove("01")

    loader = Constant(data_dir / project_folder / "rawdata" / "SubjectData.csv")

    session_updater = ppd.UpdateColumnValues(
        column_name="sessionID", func=lambda entry: entry.split("-")[1]
    )
    subject_updater = ppd.UpdateColumnValues(
        column_name="subject", func=lambda entry: entry.split("-")[1]
    )

    prep_pipe = subject_updater + session_updater

    if subject_subset is not None:
        prep_pipe += ppd.DfIsInFilter("subject", subject_subset)

    pipe = loader + ppd.CsvReader() + prep_pipe

    if pilot_pipe is None:
        if index_by_session and len(subject_subset) == 1:
            pipe += ppd.IndexSetter("sessionID", drop=True)
        return pipe

    pilot_pipe += ppd.DfInsert(column="subject", value="01")

    return BranchingPipeline(
        branches=[pilot_pipe, pipe],
        merger=lambda dfs: pd.concat(dfs, join="inner", ignore_index=True),
    )


def FoldersSelector(
    derivative,
    subject_subset=None,
    session_subset=None,
):
    """Create pipeline to load maternal sessions folder names.

    Assumes the following folder structure:
    - <data_dir>
        - maternal_brain_project
        - maternal_brain_project_pilot

    For each of the projects folder assumes:
    - <project_folder>
        - derivatives
            - <derivative_1>
            - <derivative_2>

    For each of the tools folder assumes:
    - <tool>
        - <session-folder>
        - ...

    Parameters
    ----------
    derivative : str
        Derivative folder starting (e.g. "fsl_first", "fastsurfer-long").
    subject_subset : array-like
        Id of the subjects. If None, assumes all.
        One of the following: "01", "1001B", "1004B".
    session_subset : array-like
        Subset of sessions to load. If `None`, loads all.

    Returns
    -------
    pipe : Pipeline
        Pipeline to load maternal sessions folder names.
    """
    project_folder = "maternal_brain_project"

    if subject_subset is None:
        subject_subset = MATERNAL_IDS

    if (
        session_subset is not None
        and len(subject_subset) > 1
        and "01" in subject_subset
    ):
        raise ValueError("Can't filter sessions if pilot included")

    for subject_id in subject_subset:
        if subject_id not in MATERNAL_IDS:
            raise ValueError(
                f"Oops, `{subject_id}` is not available. Please, choose from: {','.join(MATERNAL_IDS)}"
            )

    if "01" in subject_subset and len(subject_subset) > 1:
        subject_subset = list(subject_subset)
        subject_subset.remove("01")

        subject_subsets = [["01"], subject_subset]

    else:
        subject_subsets = [subject_subset]

    def _prepend_data_dir(project_folder):
        return lambda x: os.path.join(x, project_folder, "derivatives")

    pipes = []
    for subject_subset in subject_subsets:
        pilot = True if "01" in subject_subset else False

        project_folder_ = project_folder
        if pilot:
            project_folder_ += "_pilot"
            sorter = True
        else:
            sorter = _session_sorter

        pipe = (
            _prepend_data_dir(project_folder_)
            + ExpandUser()
            + FileFinder(rules=StartsWith(derivative))
            + DerivativeFoldersSelector(
                subject_subset, session_subset=session_subset, session_sorter=sorter
            )
        )

        if pilot:
            # same session metadata as 26
            pipe += ppdict.DictMap(
                ExceptionToWarning(ppdict.RemoveKeys(keys=[27]), warn=False)
            )

        pipes.append(pipe)

    if len(pipes) == 1:
        return pipes[0]

    pipe = BranchingPipeline(pipes, merger=lambda x: x[0] | x[1])

    return pipe


def MeshLoader(
    derivative,
    data_dir="~/.herbrain/data/maternal",
    subject_subset=None,
    session_subset=None,
    struct_subset=None,
    as_mesh=False,
):
    """Create pipeline to load maternal mesh filenames.

    Check out https://fsl.fmrib.ox.ac.uk/fsl/docs/#/structural/first.

    Parameters
    ----------
    derivative : str
        Derivative folder starting (e.g. "fsl_first", "fastsurfer-long").
    data_dir : str
        Directory where data is stored.
    subject_id : str
        Identification of the subject. If None, assumes pilot.
        One of the following: "01", "1001B", "1004B".
    struct_subset : str
        One of the following: 'Thal', 'Caud', 'Puta', 'Pall',
        'BrStem', 'Hipp', 'Amyg', 'Accu'.
        Suffixed with 'L_' or 'R_' (except 'BrStem').
    left : bool
        Whether to load left side. Not applicable to 'BrStem'.
    subset : array-like
        Subset of sessions to load. If `None`, loads all.

    Returns
    -------
    pipe : Pipeline
        Pipeline whose output is a nested dict whose keys are
        subject_id, session_id, struct_id.
        Values are filename or mesh.
    """
    folders_selector = Constant(data_dir) + FoldersSelector(
        subject_subset=subject_subset,
        session_subset=session_subset,
        derivative=derivative,
    )

    mesh_finder = FslMeshLoader(struct_subset, derivative, as_mesh=as_mesh)

    return folders_selector + ppdict.NestedDictMap(mesh_finder)


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

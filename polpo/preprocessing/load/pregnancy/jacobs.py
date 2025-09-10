import logging
import os
import re
from pathlib import Path

import polpo.preprocessing.dict as ppdict
import polpo.preprocessing.pd as ppd
from polpo.preprocessing import BranchingPipeline, Constant
from polpo.preprocessing.load.bids import DerivativeFoldersSelector
from polpo.preprocessing.load.fsl import (
    MeshLoader as FslMeshLoader,
)
from polpo.preprocessing.load.fsl import (
    SegmentationsLoader as FslSegmentationsLoader,
)
from polpo.preprocessing.mri import MriImageLoader
from polpo.preprocessing.path import ExpandUser, FileFinder
from polpo.preprocessing.str import DigitFinder, StartsWith

from .pilot import TabularDataLoader as PilotTabularDataLoader

MATERNAL_IDS = {"01", "1001B", "1004B"}


def _session_sorter(session_id):
    # for session_id other than in pilot
    return (
        re.sub(r"\d+$", "", session_id),
        DigitFinder(index=-1)(session_id),
    )


def TabularDataLoader(data_dir="~/.herbrain/data/maternal", subject_id=None):
    """Create pipeline to load maternal csv data.

    Parameters
    ----------
    data_dir : str
        Data root dir.
    subject_id : str
        Identification of the subject. If None, loads full dataframe (except "01").
        One of the following: "01", "1001B", "1004B".

    Returns
    -------
    pipe : Pipeline
        Pipeline to load maternal csv data.
    """
    project_folder = "maternal_brain_project"
    data_dir = Path(data_dir).expanduser()

    if subject_id == "01":
        project_folder = f"{project_folder}_pilot"

        if subject_id is not None and subject_id != "01":
            logging.warning("`subject_id` is ignored, as there's only one subject")

        return PilotTabularDataLoader(
            data_dir=data_dir / project_folder / "rawdata",
        )

    else:
        loader = Constant(data_dir / project_folder / "rawdata" / "SubjectData.csv")

        session_updater = ppd.UpdateColumnValues(
            column_name="sessionID", func=lambda entry: entry.split("-")[1]
        )
        if subject_id is not None:
            prep_pipe = (
                ppd.DfIsInFilter("subject", [f"sub-{subject_id}"])
                + ppd.Drop("subject", axis=1)
                + session_updater
                + ppd.IndexSetter(key="sessionID", drop=True)
            )
        else:
            prep_pipe = (
                ppd.UpdateColumnValues(
                    column_name="subject", func=lambda entry: entry.split("-")[1]
                )
                + session_updater
            )

    return loader + ppd.CsvReader() + prep_pipe


def FoldersSelector(
    subject_subset=None,
    session_subset=None,
    derivative="fsl",
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
    subject_subset : array-like
        Id of the subjects. If None, assumes all.
        One of the following: "01", "1001B", "1004B".
    session_subset : array-like
        Subset of sessions to load. If `None`, loads all.
    derivative : str
        Derivative folder starting (e.g. "fsl_first", "fastsurfer-long").

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

        pipes.append(pipe)

    if len(pipes) == 1:
        return pipes[0]

    pipe = BranchingPipeline(pipes, merger=lambda x: x[0] | x[1])

    return pipe


def MeshLoader(
    data_dir="~/.herbrain/data/maternal",
    subject_subset=None,
    session_subset=None,
    struct_subset=None,
    derivative="fsl",
    as_mesh=False,
):
    """Create pipeline to load maternal mesh filenames.

    Check out https://fsl.fmrib.ox.ac.uk/fsl/docs/#/structural/first.

    Parameters
    ----------
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
    derivative : str
        Derivative folder starting (e.g. "fsl_first", "fastsurfer-long").

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
    data_dir="~/.herbrain/data/maternal",
    subject_subset=None,
    session_subset=None,
    as_image=False,
    tool="fsl_first",
):
    """Create pipeline to load segmented mri filenames.

    Parameters
    ----------
    data_dir : str
        Directory where to store data.
    subset : array-like
        Subset of sessions to load. If `None`, loads all.
    subject_id : str
        Identification of the subject. If None, assumes pilot.
        One of the following: "01", "1001B", "1004B".
    as_image : bool
        Whether to load file as image.
    tool : str
        Tool used to generate derivatives.
        One of the following: "fsl*", "fast*".

    Returns
    -------
    pipe : Pipeline
        Pipeline to load segmented mri filenames.
    """
    # TODO: as_mesh?
    folders_selector = Constant(data_dir) + FoldersSelector(
        subject_subset=subject_subset,
        session_subset=session_subset,
        derivative=tool,
    )

    image_selector = FslSegmentationsLoader(tool)

    if as_image:
        image_selector += MriImageLoader()

    return folders_selector + ppdict.NestedDictMap(image_selector)

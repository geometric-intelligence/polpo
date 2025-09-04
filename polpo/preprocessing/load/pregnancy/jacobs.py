import logging
import os
import re
from pathlib import Path

import polpo.preprocessing.dict as ppdict
import polpo.preprocessing.pd as ppd
from polpo.load.first import first_struct_to_enigma_id, validate_first_struct
from polpo.preprocessing import (
    Constant,
    Contains,
    ContainsAll,
    Filter,
    Map,
    Sorter,
)
from polpo.preprocessing.path import (
    ExpandUser,
    FileFinder,
    IsFileType,
    PathShortener,
)
from polpo.preprocessing.str import DigitFinder, StartsWith

from .pilot import TabularDataLoader as PilotTabularDataLoader

MATERNAL_IDS = {"01", "1001", "1004"}


def TabularDataLoader(data_dir="~/.herbrain/data/maternal", subject_id=None):
    """Create pipeline to load maternal csv data.

    Parameters
    ----------
    data_dir : str
        Data root dir.
    subject_id : str
        Identification of the subject. If None, loads full dataframe.
        One of the following: "01", "1001B", "1004B".
    pilot : bool
        Whether to load pilot data.

    Returns
    -------
    pipe : Pipeline
        Pipeline to load maternal csv data.
    """
    # TODO: homogenize B
    project_folder = "maternal_brain_project"
    data_dir = Path(data_dir).expanduser()
    pilot = subject_id is None or subject_id == "01"

    if pilot:
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
    data_dir="~/.herbrain/data/maternal",
    subject_id=None,
    subset=None,
    as_dict=False,
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
    data_dir : str
        Directory where data is stored.
    subject_id : str
        Identification of the subject. If None, assumes pilot.
        One of the following: "01", "1001", "1004".
    subset : array-like
        Subset of sessions to load. If `None`, loads all.
    as_dict : bool
        Whether to create a dictionary with session as key.
    derivative : str
        Derivative folder starting (e.g. "fsl_first", "fastsurfer-long").

    Returns
    -------
    pipe : Pipeline
        Pipeline to load maternal sessions folder names.
    """
    # TODO: can make a generic one for BIDS: https://bids.neuroimaging.io/index.html
    project_folder = "maternal_brain_project"

    if subject_id is None:
        subject_id = "01"

    if subject_id not in MATERNAL_IDS:
        raise ValueError(
            f"Oops, `{subject_id}` is not available. Please, choose from: {','.join(MATERNAL_IDS)}"
        )

    pilot = True if subject_id == "01" else False

    if pilot:
        project_folder += "_pilot"

        if subject_id != "01":
            logging.warning("`subject_id` is ignored, as there's only one subject")

        path_to_session = PathShortener() + DigitFinder(index=-1)
        sorter = Sorter()
    else:
        path_to_session = PathShortener() + [lambda path: path.split("-")[-1]]
        sorter = Sorter(
            lambda x: (
                re.sub(r"\d+$", "", path_to_session(x)),
                DigitFinder(index=-1)(x),
            )
        )

    folder_name = os.path.join(data_dir, project_folder, "derivatives")
    folders_selector = (
        Constant(folder_name)
        + ExpandUser()
        + FileFinder(rules=StartsWith(derivative))
        + FileFinder(rules=ContainsAll([subject_id, "ses"]), as_list=True)
    )

    if subset is not None:
        folders_selector += Filter(
            func=lambda folder_name: path_to_session(folder_name) in subset
        )

    pipe = folders_selector + sorter
    if as_dict:
        pipe = pipe + ppdict.HashWithIncoming(key_step=Map(path_to_session))

    return pipe


def MeshLoader(
    data_dir="~/.herbrain/data/maternal",
    subject_id=None,
    struct="Hipp",
    subset=None,
    left=True,
    as_dict=False,
    derivative="fsl",
):
    """Create pipeline to load maternal mesh filenames.

    Check out https://fsl.fmrib.ox.ac.uk/fsl/docs/#/structural/first.

    Parameters
    ----------
    data_dir : str
        Directory where data is stored.
    subject_id : str
        Identification of the subject. If None, assumes pilot.
        One of the following: "01", "1001", "1004".
    struct : str
        One of the following: 'Thal', 'Caud', 'Puta', 'Pall',
        'BrStem', 'Hipp', 'Amyg', 'Accu'.
    left : bool
        Whether to load left side. Not applicable to 'BrStem'.
    subset : array-like
        Subset of sessions to load. If `None`, loads all.
    as_dict : bool
        Whether to create a dictionary with session as key.
    derivative : str
        Derivative folder starting (e.g. "fsl_first", "fastsurfer-long").

    Returns
    -------
    pipe : Pipeline
        Pipeline whose output is list[str] or dict[int, str].
        String represents filename. Sorting is always temporal.
    """
    folders_selector = FoldersSelector(
        data_dir=data_dir,
        subject_id=subject_id,
        subset=subset,
        as_dict=True,
        derivative=derivative,
    )

    validate_first_struct(struct)

    if struct == "BrStem":
        suffixed_side = ""
    else:
        suffixed_side = "L_" if left else "R_"

    suffixed_struct = f"{suffixed_side}{struct}"
    if derivative.startswith("enigma"):
        enigma_index = f"_{first_struct_to_enigma_id(suffixed_struct)}"
        rules = [
            lambda file: file.endswith(enigma_index),
        ]
    else:
        rules = [
            IsFileType("vtk"),
            lambda filename: suffixed_struct in filename,
        ]

    file_finder = folders_selector + ppdict.DictMap(FileFinder(rules=rules))

    if as_dict:
        return file_finder

    return file_finder + ppdict.DictToValuesList()


def SegmentationsLoader(
    data_dir="~/.herbrain/data/maternal",
    subset=None,
    subject_id=None,
    as_dict=False,
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
        One of the following: "01", "1001", "1004".
    as_dict : bool
        Whether to create a dictionary with session as key.
    tool : str
        Tool used to generate derivatives.
        One of the following: "fsl*", "fast*".

    Returns
    -------
    pipe : Pipeline
        Pipeline to load segmented mri filenames.
    """
    folders_selector = FoldersSelector(
        data_dir=data_dir,
        subject_id=subject_id,
        subset=subset,
        as_dict=True,
        derivative=tool,
    )

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

    file_finder = folders_selector + ppdict.DictMap(image_selector)

    if as_dict:
        return file_finder

    return file_finder + ppdict.DictToValuesList()

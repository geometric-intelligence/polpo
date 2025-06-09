import logging
import os
import re
from pathlib import Path

import polpo.preprocessing.pd as ppd
from polpo.preprocessing import (
    Constant,
    Contains,
    ContainsAll,
    Filter,
    Map,
    PartiallyInitializedStep,
    Sorter,
    StepWithLogging,
    TupleWith,
)
from polpo.preprocessing.dict import (
    DictMap,
    DictToValuesList,
    Hash,
    HashWithIncoming,
    SelectKeySubset,
)
from polpo.preprocessing.path import (
    ExpandUser,
    FileFinder,
    IsFileType,
    PathShortener,
)
from polpo.preprocessing.str import DigitFinder, StartsWith

from ._load import FigshareDataLoader, _get_basename

PREGNANCY_PILOT_REFLECTED_KEYS = {
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
}

FIRST_STRUCTS = {
    "Thal",
    "Caud",
    "Puta",
    "Pall",
    "BrStem",
    "Hipp",
    "Amyg",
    "Accu",
}

ENIGMA_STRUCT2ID = {
    "L_Thal": 10,
    "L_Caud": 11,
    "L_Puta": 12,
    "L_Pall": 13,
    "L_Hipp": 17,
    "L_Amyg": 18,
    "L_Accu": 26,
    "R_Thal": 49,
    "R_Caud": 50,
    "R_Puta": 51,
    "R_Pall": 52,
    "R_Hipp": 53,
    "R_Amyg": 54,
    "R_Accu": 58,
}


MATERNAL_IDS = {"01", "1001", "1004"}


class FigsharePregnancyDataLoader:
    """Transfer pregnancy data from figshare with guard rails.

    Check out
    https://figshare.com/articles/dataset/pregnancy-data/28339535
    """

    def __new__(
        cls,
        remote_path,
        data_dir=None,
        use_cache=True,
        local_basename=None,
        remove_id=True,
        validate=True,
    ):
        """Instantiate figshare data loader.

        Parameters
        ----------
        remote_path : str
            Path to retrieve from remote host.
        data_dir : str
            Directory where to store data.
        use_cache : bool
            Whether to verify if data is already available locally.
        local_basename : str
            Basename of transferred file/folder if different from remote host.
        remove_id : bool
            Whether to remove figshare added id when downloading items that are
            within a folder.
        validate : bool
            Whether to check validity of remote path.
        """
        if validate and "." not in _get_basename(remote_path):
            cls._validate_remote_path(remote_path)

        return FigshareDataLoader(
            figshare_id=28339535,
            remote_path=remote_path,
            data_dir=data_dir,
            use_cache=use_cache,
            local_basename=local_basename,
            version=1,
            remove_id=remove_id,
        )

    @staticmethod
    def _validate_remote_path(remote_path):
        def _validate_digits(string, min_value, max_value, index=0, exclude=()):
            digits = re.findall(r"\d+", string)
            if len(digits) < index + 1:
                return False

            digit = int(digits[index])
            return min_value <= digit <= max_value or digit in exclude

        valid_remote_paths = {
            "registration": lambda folder_name: folder_name
            in (
                "elastic_20250105_90",
                "elastic_20250106_80",
                "deformetrica_20250108",
            ),
            "mri": lambda folder_name: folder_name.startswith("ses-")
            and _validate_digits(
                folder_name,
                min_value=1,
                max_value=26,
            ),
            "Segmentations": lambda folder_name: folder_name.startswith("BB")
            and _validate_digits(
                folder_name,
                min_value=1,
                max_value=26,
                exclude=(15,),
            ),
        }

        remote_path_ls = remote_path.split(os.path.sep)

        if 1 <= len(remote_path_ls) <= 2 and (
            (len(remote_path_ls) == 1 and remote_path_ls[0] in valid_remote_paths)
            or valid_remote_paths[remote_path_ls[0]](remote_path_ls[1])
        ):
            return

        raise ValueError(f"{remote_path} does not exist.")


def _FigsharePregnancyFolderLoader(
    subset,
    data_dir,
    remote_path,
    id_to_path,
    none_to_subset,
    thresh=4,
    local_basename=None,
):
    """Create pipeline to load folders.

    Parameters
    ----------
    subset : array-like
        Subset of sessions to load. If `None`, loads all.
    data_dir : str
        Directory where to store data.
    remote_dir : str
        Remote directory where data is stored.
    id_to_path : callable
        Maps a session id to basename.
    local_basename : str
        Basename of transferred file/folder if different from remote host.
    none_to_subset : callable
        Creates subset with all session ids.
    thresh : int
        Sets the minimum number of folders required to download all instead.
        It will be faster than downloading individual files.

    Returns
    -------
    pipe : Pipeline
        Pipeline whose output is dict[int, string].
        Key represents session id and value the corresponding filename.
    """
    if local_basename is None:
        local_basename = remote_path

    data_dir = ExpandUser()(data_dir)
    local_dir = f"{data_dir}/{local_basename}"

    paths_to_ids = Map([PathShortener(), DigitFinder(index=0)])

    if subset is None and not os.path.exists(local_dir):
        files_loader = (
            FigsharePregnancyDataLoader(
                data_dir=data_dir,
                remote_path=remote_path,
                local_basename=local_basename,
            )
            + FileFinder()
            + Sorter()
        )

    else:
        if subset is None:
            subset = none_to_subset()

        files = FileFinder(as_list=True, warn=False)(local_dir)
        missing_subset = set(subset) - set(paths_to_ids(files))

        if len(missing_subset) == 0:
            files_loader = (
                StepWithLogging(
                    Constant(files),
                    msg=f"Data has already been downloaded... using cached file ('{local_dir}').",
                )
                + Sorter()
            )

        elif len(subset) <= thresh or len(missing_subset) <= thresh:
            files_loader = (
                Constant(subset)
                + Map(
                    PartiallyInitializedStep(
                        FigsharePregnancyDataLoader,
                        data_dir=data_dir,
                        _remote_path=lambda session_id: f"{remote_path}/{id_to_path(session_id)}",
                        _local_basename=lambda session_id: f"{local_basename}/{id_to_path(session_id)}",
                        validate=False,
                    )
                )
                + Sorter()
            )

        else:
            files_loader = (
                FigsharePregnancyDataLoader(
                    data_dir=data_dir,
                    local_basename=local_basename,
                    remote_path=remote_path,
                    use_cache=False,
                )
                + FileFinder()
                + Sorter()
            )

    return (
        files_loader + HashWithIncoming(key_step=paths_to_ids) + SelectKeySubset(subset)
    )


def PregnancyPilotMriLoader(
    subset=None, data_dir="~/.herbrain/data/pregnancy", as_dict=False
):
    """Create pipeline to load mri filenames.

    Parameters
    ----------
    subset : array-like
        Subset of sessions to load. If `None`, loads all.
    data_dir : str
        Directory where to store data.
    as_dict : bool
        Whether to create a dictionary with session as key.

    Returns
    -------
    pipe : Pipeline
        Pipeline whose output is list[str] or dict[int, str].
        String represents filename. Sorting is by session id.
    """
    folders_selector = _FigsharePregnancyFolderLoader(
        subset,
        data_dir,
        remote_path="mri",
        local_basename="raw/mri",
        id_to_path=lambda session_id: f"ses-{str(session_id).zfill(2)}",
        none_to_subset=lambda: list(range(1, 27)),
    )

    files_selector = DictMap(
        step=FileFinder(
            rules=[
                StartsWith(value="BrainNormalized"),
                IsFileType("nii.gz"),
            ]
        )
    )

    pipe = folders_selector + files_selector

    if as_dict:
        return pipe

    return pipe + DictToValuesList()


def PregnancyPilotSegmentationsLoader(
    subset=None, data_dir="~/.herbrain/data/pregnancy", as_dict=False
):
    """Create pipeline to load segmented mri filenames.

    Parameters
    ----------
    subset : array-like
        Subset of sessions to load. If `None`, loads all.
    data_dir : str
        Directory where to store data.
    as_dict : bool
        Whether to create a dictionary with session as key.

    Returns
    -------
    filenames : list[str] or dict[int, str]
        String represents filename. Sorting is by session id.
    """
    folders_selector = _FigsharePregnancyFolderLoader(
        subset,
        data_dir,
        remote_path="Segmentations",
        local_basename="derivatives/segmentations",
        id_to_path=lambda session_id: f"BB{str(session_id).zfill(2)}",
        none_to_subset=lambda: list(range(1, 15)) + list(range(16, 27)),
    )

    left_file_selector = FileFinder(
        rules=[
            StartsWith(value="left"),
            IsFileType("nii.gz"),
        ]
    )

    right_file_selector = FileFinder(
        rules=[
            StartsWith(value="right"),
            IsFileType("nii.gz"),
        ]
    )

    files_selector = DictMap(
        step=left_file_selector,
        special_step=right_file_selector,
        special_keys=PREGNANCY_PILOT_REFLECTED_KEYS,
    )

    pipe = folders_selector + files_selector

    if as_dict:
        return pipe

    return pipe + DictToValuesList()


def PregnancyPilotRegisteredMeshesLoader(
    subset=None,
    data_dir="~/.herbrain/data/pregnancy",
    as_dict=False,
    method="deformetrica",
    version=0,
):
    """Create pipeline to load registered meshes filenames.

    NB: all meshes need to be downloaded due to figshare constraints.

    Parameters
    ----------
    subset : array-like
        Subset of sessions to load. If `None`, loads all.
    data_dir : str
        Directory where to store data.
    as_dict : bool
        Whether to create a dictionary with session as key.
    method : str
        Which meshes to load based on registration methodology.
        Available options are 'deformetrica' and 'elastic'.
    version : str
        Which version of meshes to load.

    Returns
    -------
    pipe : Pipeline
        Pipeline whose output is list[str] or dict[int, str].
        String represents filename. Sorting is by session id.
    """
    method_to_path = {
        ("deformetrica", 0): "deformetrica_20250108",
        ("elastic", 0): "elastic_20250105_90",
        ("elastic", 1): "elastic_20250106_80",
    }
    path = method_to_path[(method, version)]

    local_basename = f"derivatives/{method}"
    if method == "elastic":
        local_basename = f"{local_basename}_{version}"

    paths_to_ids = Map([PathShortener(), DigitFinder(index=0)])

    pipe = (
        FigsharePregnancyDataLoader(
            data_dir="~/.herbrain/data/pregnancy",
            remote_path=f"registration/{path}",
            local_basename=local_basename,
        )
        + FileFinder(
            rules=[
                StartsWith(value="left_"),
                IsFileType("ply"),
            ],
        )
        + Sorter()
        + HashWithIncoming(key_step=paths_to_ids)
        + SelectKeySubset(subset)
    )

    if as_dict:
        return pipe

    return pipe + DictToValuesList()


def DenseMaternalCsvDataLoader(
    data_dir="~/.herbrain/data/maternal", subject_id=None, pilot=False
):
    """Create pipeline to load maternal csv data.

    Parameters
    ----------
    data_dir : str
        Data root dir.
    subject_id : str
        Identification of the subject. If None, loads full dataframe.
        One of the following: "01", "1001", "1004".
    pilot : bool
        Whether to load pilot data.

    Returns
    -------
    pipe : Pipeline
        Pipeline to load maternal csv data.
    """
    if pilot:
        if subject_id is not None and subject_id != "01":
            logging.warning("`subject_id` is ignored, as there's only one subject")

        loader = FigsharePregnancyDataLoader(
            data_dir=f"{data_dir}/raw",
            remote_path="28Baby_Hormones.csv",
            use_cache=True,
        )
        prep_pipe = ppd.UpdateColumnValues(
            column_name="sessionID", func=lambda entry: int(entry.split("-")[1])
        ) + ppd.IndexSetter(key="sessionID", drop=True)

    else:
        project_folder = "maternal_brain_project"

        loader = Constant(
            Path(data_dir) / project_folder / "rawdata" / "SubjectData.csv"
        )

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


def DenseMaternalFoldersSelector(
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
    project_folder = "maternal_brain_project"

    if subject_id is None:
        subject_id = "01"

    if subject_id not in MATERNAL_IDS:
        raise ValueError(
            f"Ups, `{subject_id}` is not available. Please, choose from: {','.join(MATERNAL_IDS)}"
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
        pipe = pipe + HashWithIncoming(key_step=Map(path_to_session))

    return pipe


def DenseMaternalMeshLoader(
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
    if struct not in FIRST_STRUCTS:
        raise ValueError(
            f"Ups, `{struct}` is not available. Please, choose from: {','.join(FIRST_STRUCTS)}"
        )

    if struct == "BrStem":
        suffixed_side = ""
    else:
        suffixed_side = "L_" if left else "R_"

    folders_selector = DenseMaternalFoldersSelector(
        data_dir=data_dir,
        subject_id=subject_id,
        subset=subset,
        as_dict=True,
        derivative=derivative,
    )

    suffixed_struct = f"{suffixed_side}{struct}"
    if derivative.startswith("enigma"):
        enigma_index = f"_{ENIGMA_STRUCT2ID[suffixed_struct]}"
        rules = [
            lambda file: file.endswith(enigma_index),
        ]
    else:
        rules = [
            IsFileType("vtk"),
            lambda filename: suffixed_struct in filename,
        ]

    file_finder = folders_selector + DictMap(FileFinder(rules=rules))

    if as_dict:
        return file_finder

    return file_finder + DictToValuesList()


def DenseMaternalSegmentationsLoader(
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
    folders_selector = DenseMaternalFoldersSelector(
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
        raise ValueError(f"Ups, don't know how to handle: {tool}")

    file_finder = folders_selector + DictMap(image_selector)

    if as_dict:
        return file_finder

    return file_finder + DictToValuesList()


def NeuroMaternalFoldersSelector(
    data_dir="~/.herbrain/data/maternal/neuromaternal_madrid_2021",
    subset=None,
    derivative="enigma",
):
    # TODO: homogenize data_dir with Pregnancy one; use derivatives folder above
    # TODO: bring as_dict in?
    # TODO: as dict in session is particularly interesting
    # TODO: subset applyes to subject id

    # per subject, per session

    path_to_sub = PathShortener() + (lambda x: x.split("_")[0].split("-")[-1])

    # BranchingPipeline(
    #     [DigitFinder(index=0), DigitFinder(index=-1)], merger=lambda x: x
    # )

    folder_name = os.path.join(data_dir, "derivatives")
    folders_selector = (
        Constant(folder_name)
        + ExpandUser()
        + FileFinder(rules=StartsWith(derivative))
        + FileFinder(rules=ContainsAll(["sub", "ses"]), as_list=True)
    )

    if subset is not None:
        folders_selector += Filter(
            func=lambda folder_name: path_to_sub(folder_name) in subset
        )

    pipe = (
        folders_selector
        + TupleWith(Map(path_to_sub), incoming_first=False)
        + Sorter(key=lambda x: x[0])
        + (
            lambda sub_sessions: [
                (x[0], (x[1], y[1]))
                for x, y in zip(sub_sessions[::2], sub_sessions[1::2])
                if x[0] == y[0]
            ]
        )
        + Hash()
        + DictMap(step=Sorter(key=lambda x: x.split()[-1]))
    )

    return pipe


def NeuroMaternalMeshLoader(
    data_dir="~/.herbrain/data/maternal/neuromaternal_madrid_2021",
    subject_id=None,
    struct="Hipp",
    subset=None,
    left=True,
    as_dict=False,
    derivative="enigma",
):
    # NB: as_dict controls session

    if struct not in FIRST_STRUCTS:
        raise ValueError(
            f"Ups, `{struct}` is not available. Please, choose from: {','.join(FIRST_STRUCTS)}"
        )

    if struct == "BrStem":
        suffixed_side = ""
    else:
        suffixed_side = "L_" if left else "R_"

    folders_selector = NeuroMaternalFoldersSelector(
        data_dir=data_dir,
        subset=subset,
        derivative=derivative,
    )

    suffixed_struct = f"{suffixed_side}{struct}"
    if derivative.startswith("enigma"):
        enigma_index = f"_{ENIGMA_STRUCT2ID[suffixed_struct]}"
        rules = [
            lambda file: file.endswith(enigma_index),
        ]
    else:
        rules = [
            IsFileType("vtk"),
            lambda filename: suffixed_struct in filename,
        ]

    path_to_session = PathShortener() + DigitFinder(index=-1)

    # NB: sessions are already sorted
    if as_dict:
        file_finder = DictMap(
            HashWithIncoming(
                Map(FileFinder(rules=rules)),
                key_step=Map(path_to_session),
            )
        )
    else:
        file_finder = DictMap(Map(FileFinder(rules=rules)))

    pipe = folders_selector + file_finder

    return pipe

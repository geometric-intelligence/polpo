import logging
import os
import re
from pathlib import Path

import polpo.preprocessing.dict as ppdict
import polpo.preprocessing.pd as ppd
from polpo.preprocessing import (
    BranchingPipeline,
    CachablePipeline,
    Constant,
    Contains,
    ContainsAll,
    EnsureIterable,
    Filter,
    IdentityStep,
    IndexMap,
    Map,
    PartiallyInitializedStep,
    Sorter,
    StepWithLogging,
    TupleWith,
)
from polpo.preprocessing.mesh.conversion import PvFromData
from polpo.preprocessing.mesh.io import FreeSurferReader, PvReader, PvWriter
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


def _tool_to_mesh_reader(tool):
    # update for other tools
    if tool.startswith("enigma"):
        return FreeSurferReader() + PvFromData()
    else:
        raise ValueError(f"Oops, don't know how to handle: {tool}")


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
        files_loader
        + ppdict.HashWithIncoming(key_step=paths_to_ids)
        + ppdict.SelectKeySubset(subset)
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

    files_selector = ppdict.DictMap(
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

    return pipe + ppdict.DictToValuesList()


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

    files_selector = ppdict.DictMap(
        step=left_file_selector,
        special_step=right_file_selector,
        special_keys=PREGNANCY_PILOT_REFLECTED_KEYS,
    )

    pipe = folders_selector + files_selector

    if as_dict:
        return pipe

    return pipe + ppdict.DictToValuesList()


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
        + ppdict.HashWithIncoming(key_step=paths_to_ids)
        + ppdict.SelectKeySubset(subset)
    )

    if as_dict:
        return pipe

    return pipe + ppdict.DictToValuesList()


def DenseMaternalCsvDataLoader(data_dir="~/.herbrain/data/maternal", subject_id=None):
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
    project_folder = "maternal_brain_project"
    data_dir = Path(data_dir).expanduser()
    pilot = subject_id is None or subject_id == "01"

    if pilot:
        project_folder = f"{project_folder}_pilot"

        if subject_id is not None and subject_id != "01":
            logging.warning("`subject_id` is ignored, as there's only one subject")

        loader = FigsharePregnancyDataLoader(
            data_dir=data_dir / project_folder / "rawdata",
            remote_path="28Baby_Hormones.csv",
            use_cache=True,
        )
        prep_pipe = ppd.UpdateColumnValues(
            column_name="sessionID", func=lambda entry: int(entry.split("-")[1])
        ) + ppd.IndexSetter(key="sessionID", drop=True)

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
            f"Oops, `{struct}` is not available. Please, choose from: {','.join(FIRST_STRUCTS)}"
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

    file_finder = folders_selector + ppdict.DictMap(FileFinder(rules=rules))

    if as_dict:
        return file_finder

    return file_finder + ppdict.DictToValuesList()


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
        raise ValueError(f"Oops, don't know how to handle: {tool}")

    file_finder = folders_selector + ppdict.DictMap(image_selector)

    if as_dict:
        return file_finder

    return file_finder + ppdict.DictToValuesList()


def NeuroMaternalFoldersSelector(
    data_dir="~/.herbrain/data/maternal/neuromaternal_madrid_2021",
    subset=None,
    derivative="enigma",
    remove_missing_sessions=True,
):
    """Create pipeline to load neuromaternal mesh folders.

    Parameters
    ----------
    subset : array-like
        Subset of participants to load. If `None`, loads all.
    remove_missing_sessions : bool
        Whether to keep only subjects for which there's two sessions.

    Returns
    -------
    pipe : Pipeline
        Pipeline whose output is dict[str, list[str]].
        Key represents participant id and value the corresponding filenames.
    """
    # TODO: homogenize data_dir with Pregnancy one; use derivatives folder above

    path_to_sub = PathShortener() + (lambda x: x.split("_")[0].split("-")[-1])

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

    def _group_sessions(sub_sessions):
        out = {}
        for sub_session in sub_sessions:
            key = sub_session[0]
            sub_out = out.get(key, [])
            sub_out.append(sub_session[1])
            out[key] = sub_out

        return out

    filter_ = (
        # TODO: this get rid of 3's; bring them in again
        ppdict.DictFilter(func=lambda x: len(x) == 2)
        if remove_missing_sessions
        else (lambda x: x)
    )

    pipe = (
        folders_selector
        + TupleWith(Map(path_to_sub), incoming_first=False)
        + _group_sessions
        + filter_
        + ppdict.DictMap(step=Sorter(key=lambda x: x.split()[-1]))
    )

    return pipe


def _neuromaternal_session_id_map(value):
    return value - 3


def NeuroMaternalMeshLoader(
    data_dir="~/.herbrain/data/maternal/neuromaternal_madrid_2021",
    struct="Hipp",
    subset=None,
    left=True,
    as_dict=False,
    derivative="enigma",
    as_mesh=False,
):
    # NB: as_dict controls session
    # NB: sessions names are remapped to start at zero

    # TODO: update uses of left and right?
    # TODO: update behavior of other mesh loaders

    if struct not in FIRST_STRUCTS:
        raise ValueError(
            f"Oops, `{struct}` is not available. Please, choose from: {','.join(FIRST_STRUCTS)}"
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

    path_to_session = (
        PathShortener()
        + DigitFinder(index=-1)
        + (lambda x: _neuromaternal_session_id_map(x))
    )

    # NB: sessions are already sorted
    if as_dict:
        file_finder = ppdict.DictMap(
            ppdict.HashWithIncoming(
                Map(FileFinder(rules=rules)),
                key_step=Map(path_to_session),
            )
        )
    else:
        file_finder = ppdict.DictMap(Map(FileFinder(rules=rules)))

    pipe = folders_selector + file_finder

    if not as_mesh:
        return pipe

    return pipe + ppdict.NestedDictMap(
        _tool_to_mesh_reader(derivative), inner_is_dict=as_dict, depth=1
    )


def NeuroMaternalMultiMeshLoader(
    data_dir="~/.herbrain/data/maternal/neuromaternal_madrid_2021",
    subset=None,
    as_inner_dict=False,
    derivative="enigma",
    as_mesh=False,
):
    # TODO: update name? make other private

    return EnsureIterable(
        ppdict.HashWithIncoming(
            Map(
                PartiallyInitializedStep(
                    Step=NeuroMaternalMeshLoader,
                    as_dict=as_inner_dict,
                    pass_data=False,
                    _struct=lambda name: name.split("_")[-1],
                    _left=lambda name: name.split("_")[0] == "L",
                    derivative=derivative,
                    as_mesh=as_mesh,
                )
            )
        )
    )


def NeuroMaternalTabularDataLoader(
    data_dir="~/.herbrain/data/maternal/neuromaternal_madrid_2021",
    keep_mothers=True,
    keep_control=True,
    sessions_to_keep=(0, 1),
):
    """Load neuro maternal tabular data.

    Parameters
    ----------
    data_dir : str
        Project directory.
    keep_mothers : bool
        Wether to keep mothers.
    keep_control : bool
        Whether to keep control.

    Returns
    -------
    pipe : Pipeline
    """
    filename = os.path.join(data_dir, "rawdata", "participants_long_czi.tsv")

    load_pipe = ppd.CsvReader(filename, delimiter="\t")

    prep_pipe = (
        ppd.UpdateColumnValues(
            column_name="participant_id",
            func=lambda entry: entry.split("-")[1],
        )
        + ppd.UpdateColumnValues(
            column_name="ses",
            func=lambda entry: _neuromaternal_session_id_map(int(entry.split("-")[1])),
        )
        + ppd.DfIsInFilter("ses", sessions_to_keep, readonly=False)
        + ppd.Drop(labels=["participant_id_ses"], axis=1, inplace=True)
    )

    if not keep_mothers:
        prep_pipe += ppd.DfFilter(lambda df: df["group"] == "mother", negate=True)

    if not keep_control:
        prep_pipe += ppd.DfFilter(lambda df: df["group"] == "control", negate=True)

    return load_pipe + prep_pipe


def CacheableMeshLoader(
    cache_dir,
    pipe,
    use_cache=True,
    cache=True,
    overwrite=True,
):
    # TODO: move place?
    # TODO: make reader and writer package agnostic?

    # TODO: check DictToValuesList and depth
    cache_pipe = (
        FileFinder(IsFileType(ext="vtk"))
        + Sorter()
        + ppdict.HashWithIncoming(Map(PvReader()))
        + ppdict.DictMap(key_step=lambda x: x.rsplit("/", maxsplit=1)[1].split(".")[0])
        + ppdict.NestDict(sep="-")
        + ppdict.NestedDictMap(ppdict.DictToValuesList(), depth=1)
    )

    # TODO: update depth
    # <struct>-<participant>-<mesh-index>
    to_cache_pipe = IndexMap(
        index=1,
        step=(
            ppdict.NestedDictMap(
                BranchingPipeline([lambda x: range(len(x)), IdentityStep()]) + dict,
                depth=1,
            )
            + ppdict.UnnestDict(sep="-")
        ),
    ) + (
        (lambda data: {f"{data[0]}/{key}": value for key, value in data[1].items()})
        + ppdict.DictToTuplesList()
        + Map(PvWriter(ext="vtk"))
    )

    return CachablePipeline(
        cache_dir,
        pipe,
        cache_pipe,
        to_cache_pipe,
        use_cache=use_cache,
        cache=cache,
        overwrite=overwrite,
    )

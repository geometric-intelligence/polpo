"""Load pipeline for pilot data.

Data is available at
https://figshare.com/articles/dataset/pregnancy-data/28339535.

Segmentations refer to hippocampal subfields.
"""

import os
import re

import polpo.preprocessing.dict as ppdict
import polpo.preprocessing.pd as ppd
from polpo.preprocessing import (
    Constant,
    Map,
    PartiallyInitializedStep,
    Sorter,
    StepWithLogging,
)
from polpo.preprocessing.load.figshare import FigshareDataLoader, _get_basename
from polpo.preprocessing.mri import MriImageLoader
from polpo.preprocessing.path import (
    ExpandUser,
    FileFinder,
    IsFileType,
    PathShortener,
)
from polpo.preprocessing.str import DigitFinder, StartsWith

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


# TODO: add the possibility to download and reorganize data
# then use loaders available in maternal


class FigsharePregnancyDataLoader:
    """Transfer pregnancy data from figshare with guard rails.

    Check out
    https://figshare.com/articles/dataset/pregnancy-data/28339535
    """

    # TODO: make private?
    # TODO: rename?

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
    # TODO: rename?
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


def TabularDataLoader(data_dir="~/.herbrain/data/pregnancy", use_cache=True):
    loader = FigsharePregnancyDataLoader(
        data_dir=data_dir,
        remote_path="28Baby_Hormones.csv",
        use_cache=use_cache,
    )

    prep_pipe = ppd.UpdateColumnValues(
        column_name="sessionID", func=lambda entry: int(entry.split("-")[1])
    ) + ppd.IndexSetter(key="sessionID", drop=True)

    return loader + ppd.CsvReader() + prep_pipe


def MriLoader(
    subset=None,
    data_dir="~/.herbrain/data/pregnancy",
    as_image=False,
):
    """Create pipeline to load mri filenames.

    Parameters
    ----------
    subset : array-like
        Subset of sessions to load. If `None`, loads all.
    data_dir : str
        Directory where to store data.
    as_image : bool
        Whether to load file as image.

    Returns
    -------
    pipe : Pipeline
        Pipeline whose output is dict[int, str or np.array].
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
    if as_image:
        return pipe + ppdict.DictMap(MriImageLoader())

    return pipe


def HippocampalSubfieldsSegmentationsLoader(
    subset=None,
    data_dir="~/.herbrain/data/pregnancy",
    as_image=False,
):
    """Create pipeline to load segmented mri filenames.

    Parameters
    ----------
    subset : array-like
        Subset of sessions to load. If `None`, loads all.
    data_dir : str
        Directory where to store data.
    as_image : bool
        Whether to load file as image.

    Returns
    -------
    filenames : dict[int, str]
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

    if as_image:
        return pipe + ppdict.DictMap(MriImageLoader())

    return pipe


def RegisteredMeshesLoader(
    subset=None,
    data_dir="~/.herbrain/data/pregnancy",
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
    method : str
        Which meshes to load based on registration methodology.
        Available options are 'deformetrica' and 'elastic'.
    version : str
        Which version of meshes to load.

    Returns
    -------
    pipe : Pipeline
        Pipeline whose output is dict[int, str].
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

    return pipe

import os
import re

from polpo.preprocessing import (
    BranchingPipeline,
    IfCondition,
    IndexSelector,
    Map,
    Sorter,
)
from polpo.preprocessing.dict import (
    DictKeysFilter,
    DictToTuplesList,
    Hash,
    HashWithIncoming,
)
from polpo.preprocessing.path import FileFinder, FileRule, IsFileType, PathShortener
from polpo.preprocessing.str import DigitFinder

from ._load import FigshareDataLoader, _get_basename

PREGNANCY_PILOT_REFLECTED_KEYS = (
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
)


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
        """
        if "." not in _get_basename(remote_path):
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
    """
    folders_selector = (
        FigsharePregnancyDataLoader(
            data_dir=data_dir,
            remote_path="Segmentations",
        )
        + FileFinder()
        + Sorter()
    ) + HashWithIncoming(
        key_step=Map([PathShortener(), DigitFinder(index=0)]),
    )

    if subset is not None:
        folders_selector = folders_selector + DictKeysFilter(values=subset)

    folders_selector = folders_selector + DictToTuplesList()

    left_file_selector = FileFinder(
        rules=[
            FileRule(value="left", func="startswith"),
            IsFileType("nii.gz"),
        ]
    )

    right_file_selector = FileFinder(
        rules=[
            FileRule(value="right", func="startswith"),
            IsFileType("nii.gz"),
        ]
    )

    file_selector = IfCondition(
        step=IndexSelector(1) + left_file_selector,
        else_step=IndexSelector(1) + right_file_selector,
        condition=lambda datum: datum[0] not in PREGNANCY_PILOT_REFLECTED_KEYS,
    )

    if as_dict:
        return (
            folders_selector
            + BranchingPipeline(
                [
                    Map(IndexSelector(0)),
                    Map(file_selector),
                ]
            )
            + Hash()
        )

    return folders_selector + Map(file_selector)

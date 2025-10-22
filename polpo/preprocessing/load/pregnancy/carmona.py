import os

import polpo.preprocessing.dict as ppdict
import polpo.preprocessing.pd as pppd
from polpo.preprocessing import Constant
from polpo.preprocessing.load.bids import DerivativeFoldersSelector
from polpo.preprocessing.load.fsl import MeshLoader as FslMeshLoader
from polpo.preprocessing.path import ExpandUser, FileFinder
from polpo.preprocessing.str import StartsWith


def FoldersSelector(
    subject_subset=None,
    session_subset=None,
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
    # this is mostly the same as jacobs
    pipe = (
        (lambda x: os.path.join(x, "derivatives"))
        + ExpandUser()
        + FileFinder(rules=StartsWith(derivative))
        + DerivativeFoldersSelector(
            subject_subset, session_subset=session_subset, session_sorter=True
        )
    )

    if remove_missing_sessions:
        pipe += ppdict.DictFilter(func=lambda x: len(x) == 2)

    return pipe


def MeshLoader(
    data_dir="~/.herbrain/data/maternal/neuromaternal_madrid_2021",
    subject_subset=None,
    session_subset=None,
    struct_subset=None,
    derivative="enigma",
    remove_missing_sessions=True,
    as_mesh=False,
):
    folders_selector = Constant(data_dir) + FoldersSelector(
        subject_subset=subject_subset,
        session_subset=session_subset,
        derivative=derivative,
        remove_missing_sessions=remove_missing_sessions,
    )

    mesh_finder = FslMeshLoader(struct_subset, derivative, as_mesh)

    return folders_selector + ppdict.NestedDictMap(mesh_finder)


def TabularDataLoader(
    data_dir="~/.herbrain/data/maternal/neuromaternal_madrid_2021",
    keep_mothers=True,
    keep_control=True,
    sessions_to_keep=(3, 4),
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

    load_pipe = pppd.CsvReader(filename, delimiter="\t")

    prep_pipe = (
        pppd.UpdateColumnValues(
            column_name="participant_id",
            func=lambda entry: entry.split("-")[1],
        )
        + pppd.UpdateColumnValues(
            column_name="ses",
            func=lambda entry: int(entry.split("-")[1]),
        )
        + pppd.DfIsInFilter("ses", sessions_to_keep, readonly=False)
        + pppd.Drop(labels=["participant_id_ses"], axis=1, inplace=True)
    )

    if not keep_mothers:
        prep_pipe += pppd.DfFilter(lambda df: df["group"] == "mother", negate=True)

    if not keep_control:
        prep_pipe += pppd.DfFilter(lambda df: df["group"] == "control", negate=True)

    return load_pipe + prep_pipe

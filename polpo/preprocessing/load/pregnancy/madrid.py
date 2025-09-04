import os

import polpo.preprocessing.dict as ppdict
import polpo.preprocessing.pd as pppd
from polpo.preprocessing import (
    Constant,
    ContainsAll,
    EnsureIterable,
    Filter,
    Map,
    PartiallyInitializedStep,
    Sorter,
    TupleWith,
)
from polpo.preprocessing.load.fsl import (
    first_struct_to_enigma_id,
    tool_to_mesh_reader,
    validate_first_struct,
)
from polpo.preprocessing.path import (
    ExpandUser,
    FileFinder,
    IsFileType,
    PathShortener,
)
from polpo.preprocessing.str import DigitFinder, StartsWith


def _neuromaternal_session_id_map(value):
    return value - 3


def FoldersSelector(
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


def MeshLoader(
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
    folders_selector = FoldersSelector(
        data_dir=data_dir,
        subset=subset,
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
        tool_to_mesh_reader(derivative), inner_is_dict=as_dict, depth=1
    )


def MultiMeshLoader(
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
                    Step=MeshLoader,
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


def TabularDataLoader(
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

    load_pipe = pppd.CsvReader(filename, delimiter="\t")

    prep_pipe = (
        pppd.UpdateColumnValues(
            column_name="participant_id",
            func=lambda entry: entry.split("-")[1],
        )
        + pppd.UpdateColumnValues(
            column_name="ses",
            func=lambda entry: _neuromaternal_session_id_map(int(entry.split("-")[1])),
        )
        + pppd.DfIsInFilter("ses", sessions_to_keep, readonly=False)
        + pppd.Drop(labels=["participant_id_ses"], axis=1, inplace=True)
    )

    if not keep_mothers:
        prep_pipe += pppd.DfFilter(lambda df: df["group"] == "mother", negate=True)

    if not keep_control:
        prep_pipe += pppd.DfFilter(lambda df: df["group"] == "control", negate=True)

    return load_pipe + prep_pipe

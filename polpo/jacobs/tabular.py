from pathlib import Path

import pandas as pd

import polpo.preprocessing.pd as ppd
import polpo.utils as putils
from polpo.preprocessing import BranchingPipeline, Constant, pipe_to_func

from .pilot.tabular import TabularDataLoader as PilotTabularDataLoader


def _remove_prefix(entry, sep="-"):
    return entry.split(sep)[1]


def _TabularDataLoader(
    data_dir="~/.herbrain/data/maternal/maternal_brain_project/rawdata",
    subject_subset=None,
    index_by_session=False,
):
    """Create pipeline to load maternal csv data.

    Parameters
    ----------
    data_dir : str
        Data root dir.
    subject_subset : array-like
        Id of the subjects. If None, assumes all.
        If pilot and other, loads only common columns.
    index_by_session : bool
        Whether to index the dataframe by session.
        Only applies if one subject.

    Returns
    -------
    pipe : Pipeline
        Pipeline to load maternal csv data without the pilot.
    """
    filename = "SessionData.csv"
    loader = Constant(Path(data_dir).expanduser() / filename)

    prep_pipe = ppd.UpdateColumnValues(
        column_name="sessionID",
        func=_remove_prefix,
    )
    prep_pipe += ppd.UpdateColumnValues(
        column_name="subject",
        func=_remove_prefix,
    )

    if subject_subset is not None:
        prep_pipe += ppd.DfIsInFilter("subject", subject_subset)

    if index_by_session and (subject_subset is not None and len(subject_subset) == 1):
        prep_pipe += ppd.IndexSetter("sessionID", drop=True)

    return loader + ppd.CsvReader() + prep_pipe


def TabularDataLoader(
    data_dir="~/.herbrain/data/maternal",
    subject_subset=None,
    index_by_session=False,
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
    data_dir = Path(data_dir).expanduser()
    project_folder = "maternal_brain_project"

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

    pipe = _TabularDataLoader(
        data_dir=data_dir / project_folder / "rawdata",
        subject_subset=subject_subset,
        index_by_session=index_by_session and len(subject_subset) == 1,
    )

    if pilot_pipe is None and len(subject_subset) == 1:
        return pipe

    pilot_pipe += ppd.DfInsert(column="subject", value="01")

    return BranchingPipeline(
        branches=[pilot_pipe, pipe],
        merger=lambda dfs: pd.concat(dfs, join="inner", ignore_index=True),
    )


def get_key_to_week(
    data_dir="~/.herbrain/data/maternal",
    subject_subset=None,
):
    df = TabularDataLoader(data_dir=data_dir, subject_subset=subject_subset)()

    return putils.df_to_nested_dict(
        df, outer_key="subject", inner_key="sessionID", value_col="gestWeek"
    )


get_tabular_data = pipe_to_func(TabularDataLoader)

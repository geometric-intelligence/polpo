from pathlib import Path

import polpo.preprocessing.pd as ppd
from polpo.preprocessing import Constant, pipe_to_func


def TabularDataLoader(
    data_dir="~/.herbrain/data/maternal/maternal_brain_project_pilot/rawdata",
    index_by_session=True,
    remove_repeated=True,
):
    filename = "SessionData.csv"
    loader = Constant(Path(data_dir).expanduser() / filename)

    prep_pipe = ppd.UpdateColumnValues(
        column_name="sessionID", func=lambda entry: entry.split("-")[1]
    )
    if remove_repeated:
        prep_pipe += ppd.DfFilter(lambda df: df["sessionID"] == "27", negate=True)

    if index_by_session:
        prep_pipe += ppd.IndexSetter("sessionID", drop=True)

    return loader + ppd.CsvReader() + prep_pipe


get_tabular_data = pipe_to_func(TabularDataLoader)

import os

import polpo.preprocessing.pd as pppd


def SessionDataLoader(
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

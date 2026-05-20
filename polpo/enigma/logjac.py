import numpy as np
import polars as pl

import polpo.utils as putils

from .naming import get_all_subcortical_structs, name_to_aseg_id


def _index2cols(header, index):
    return [c for c in header.columns if c.startswith(f"LogJacs_{index}_")]


def _struct_subset2cols(header, struct_subset):
    if struct_subset is None:
        struct_subset = get_all_subcortical_structs(order=True)

    enigma_indices = [name_to_aseg_id(struct) for struct in struct_subset]

    name2cols = {}
    for name, index in zip(struct_subset, enigma_indices):
        name2cols[name] = _index2cols(header, index)

    return name2cols


def load_logjac_session(filename, struct_subset=None):
    df = pl.read_csv(filename)
    name2cols = _struct_subset2cols(df, struct_subset)

    data = {}
    for name, cols in name2cols.items():
        data[name] = df[cols].to_numpy().squeeze()

    return data


def load_logjac_sessions(filenames, struct_subset=None):
    name2cols = None
    all_cols = None

    data = []
    for filename in filenames:
        df = pl.read_csv(filename, columns=all_cols)
        if name2cols is None:
            name2cols = _struct_subset2cols(df, struct_subset)
            all_cols = putils.unnest_list(name2cols.values())

        data_ = {}
        for struct_name, cols in name2cols.items():
            data_[struct_name] = df[cols].to_numpy().squeeze()

        data.append(data_)

    return data


def load_logjac_group(
    filename,
    subject_subset=None,
    session_subset=None,
    struct_subset=None,
):
    df = pl.read_csv(filename)
    name2cols = _struct_subset2cols(df, struct_subset)

    df = df.with_columns(
        pl.col("SubjID").str.extract(r"sub-([^_]+)", 1).alias("subj_id"),
        pl.col("SubjID").str.extract(r"_ses-(.+)$", 1).alias("ses_id"),
    )

    if subject_subset is not None:
        df = df.filter(pl.col("subj_id").is_in(subject_subset))

    if session_subset is not None:
        df = df.filter(pl.col("ses_id").is_in(session_subset))

    data = {}
    for row in df.iter_rows(named=True):
        subj_id = row["subj_id"]
        ses_id = row["ses_id"]

        data.setdefault(subj_id, {})[ses_id] = {
            struct_name: np.asarray([row[c] for c in cols])
            for struct_name, cols in name2cols.items()
        }

    return data

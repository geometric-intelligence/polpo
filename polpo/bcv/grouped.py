from itertools import combinations

import numpy as np


def compute_held_out_rows(group_ids, n_groups=1):
    """Compute held-out row indices from group identifiers.

    Each fold holds out all rows belonging to a combination of
    ``n_groups`` distinct groups.

    Parameters
    ----------
    group_ids : array-like
        Outer-group identifier associated with each row.
    n_groups : int
        Number of outer groups to hold out in each fold.

    Returns
    -------
    held_out_rows : dict
        Mapping from tuples of held-out outer-group identifiers to the
        corresponding row indices.
    """
    unique_group_ids = np.unique(group_ids)

    return {
        held_out_ids: np.flatnonzero(np.isin(group_ids, held_out_ids))
        for held_out_ids in combinations(unique_group_ids, n_groups)
    }


def group_ids_from_sizes(sizes, labels=None):
    """Create group identifiers from group sizes.

    Parameters
    ----------
    sizes : array-like
        Number of rows belonging to each group.
    labels : array-like
        Identifier for each group. If None, consecutive integer identifiers
        are used.

    Returns
    -------
    ndarray
        Integer outer-group identifier associated with each row.
    """
    if labels is None:
        labels = np.arange(len(sizes))

    if len(labels) != len(sizes):
        raise ValueError("labels and sizes must have the same length.")

    return np.repeat(labels, sizes)

import numpy as np


def select_rank_min_error(fold_errors):
    """Select the rank with minimum mean cross-validation error.

    Parameters
    ----------
    fold_errors : array-like, shape (n_folds, n_ranks)
        Cross-validation errors for each fold and candidate rank.

    Returns
    -------
    rank : int
        Selected rank.
    """
    mean_errors = np.mean(fold_errors, axis=0)
    return np.argmin(mean_errors) + 1


def select_rank_one_se(fold_errors):
    """Select the smallest rank within one standard error of the minimum.

    The mean cross-validation error is computed for each candidate rank.
    The selected rank is the smallest rank whose mean error is no greater
    than the minimum mean error plus one standard error evaluated at the
    minimizing rank.

    Parameters
    ----------
    fold_errors : array-like, shape (n_folds, n_ranks)
        Cross-validation errors for each fold and candidate rank.

    Returns
    -------
    rank : int
        Selected rank.
    """
    n_blocks = fold_errors.shape[0]

    mean_errors = np.mean(fold_errors, axis=0)
    std_errors = np.std(fold_errors, axis=0, ddof=1)
    se_errors = std_errors / np.sqrt(n_blocks)

    best_idx = np.argmin(mean_errors)
    threshold = mean_errors[best_idx] + se_errors[best_idx]

    selected_idx = np.flatnonzero(mean_errors <= threshold)[0]

    return selected_idx + 1


def select_rank_one_se_grouped(fold_errors):
    """Select rank using the one-standard-error rule after group aggregation.

    Errors are first averaged across dependent folds within each group.
    The mean cross-validation error and its standard error are then computed
    across groups. The selected rank is the smallest rank whose mean error is
    no greater than the minimum mean error plus one standard error evaluated
    at the minimizing rank.

    Parameters
    ----------
    fold_errors : array-like, shape (n_groups, n_folds_per_group, n_ranks)
        Cross-validation errors organized by independent groups and dependent
        folds within each group.

    Returns
    -------
    rank : int
        Selected rank.
    """
    group_errors = np.mean(fold_errors, axis=1)
    return select_rank_one_se(group_errors)

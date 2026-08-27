import numpy as np
from matplotlib import pyplot as plt


def _rank_error_stats(fold_errors, error):
    mean_errors = fold_errors.mean(axis=0)
    std_errors = fold_errors.std(axis=0, ddof=1)

    if error == "std":
        errors = std_errors
    elif error == "se":
        errors = std_errors / np.sqrt(fold_errors.shape[0])
    else:
        raise ValueError("error must be 'std' or 'se'.")

    ranks = np.arange(1, mean_errors.shape[0] + 1)
    return ranks, mean_errors, errors


def _plot_best_rank(ax, ranks, mean_errors, best_rank):
    if best_rank is None:
        return

    best_idx = best_rank - 1
    ax.scatter(
        ranks[best_idx],
        mean_errors[best_idx],
        color="red",
        zorder=3,
        label=f"Best rank: {best_rank}",
    )
    ax.legend()


def plot_rank_errors_band(
    fold_errors,
    best_rank=None,
    error="se",
    ax=None,
):
    ranks, mean_errors, errors = _rank_error_stats(fold_errors, error)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(ranks, mean_errors)
    ax.fill_between(
        ranks,
        mean_errors - errors,
        mean_errors + errors,
        alpha=0.2,
    )

    _plot_best_rank(ax, ranks, mean_errors, best_rank)

    ax.set_xlabel("Rank")
    ax.set_ylabel("BCV error")

    return ax


def plot_rank_errors_errorbar(
    fold_errors,
    best_rank=None,
    error="std",
    ax=None,
):
    ranks, mean_errors, errors = _rank_error_stats(fold_errors, error)

    if ax is None:
        _, ax = plt.subplots()

    ax.errorbar(
        ranks,
        mean_errors,
        yerr=errors,
        marker="o",
        capsize=3,
    )

    _plot_best_rank(ax, ranks, mean_errors, best_rank)

    ax.set_xlabel("Rank")
    ax.set_ylabel("BCV error")

    return ax

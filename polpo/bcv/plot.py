import numpy as np

from polpo.plot.pyplot import plot_mean_band, plot_mean_errorbar


def _plot_best_rank(ax, best_rank, line_index=-1):
    if best_rank is None:
        return

    line = ax.lines[line_index]
    x, y = line.get_data()

    index = np.where(x == best_rank)[0][0]
    ax.scatter(
        x[index],
        y[index],
        color="red",
        zorder=3,
        label=f"Best rank: {best_rank}",
    )
    ax.legend()

    return ax


def plot_rank_errors_band(
    fold_errors,
    best_rank=None,
    error="se",
    ax=None,
):
    ranks = np.arange(1, fold_errors.shape[1] + 1)

    ax = plot_mean_band(
        ranks,
        fold_errors,
        error=error,
        ax=ax,
    )
    _plot_best_rank(ax, best_rank, line_index=-1)

    ax.set_xlabel("Rank")
    ax.set_ylabel("BCV error")

    return ax


def plot_rank_errors_errorbar(
    fold_errors,
    best_rank=None,
    error="std",
    ax=None,
):
    ranks = np.arange(1, fold_errors.shape[1] + 1)

    line_index = len(ax.lines) if ax is None else 0

    ax = plot_mean_errorbar(
        ranks,
        fold_errors,
        error=error,
        ax=ax,
    )
    _plot_best_rank(ax, best_rank, line_index=line_index)

    ax.set_xlabel("Rank")
    ax.set_ylabel("BCV error")

    return ax

import numpy as np
from matplotlib import pyplot as plt

from .base import Plotter


class SlicesPlotter(Plotter):
    def __init__(self, cmap="gray", vertical=False):
        super().__init__()
        self.cmap = cmap
        self.vertical = vertical

    def plot(self, slices):
        if isinstance(slices, np.ndarray):
            n_slices = 1
            slices = [slices]
        else:
            n_slices = len(slices)

        n_rows, n_cols = 1, n_slices
        if self.vertical:
            n_rows, n_cols = n_cols, n_rows

        fig, axes = plt.subplots(n_rows, n_cols, constrained_layout=True)
        if n_slices == 1:
            axes = [axes]

        for ax, slice_ in zip(axes, slices):
            ax.imshow(
                slice_.T,
                cmap=self.cmap,
                origin="lower",
            )

        return fig, axes


def _get_groups_and_subgroups(data, has_all=True):
    # has_all: whether all groups have all subgroups
    groups = list(data.keys())

    if has_all:
        subgroups = list(data[groups[0]].keys())
    else:
        subgroups = sorted(
            {subgroup for group in data.values() for subgroup in group.keys()}
        )

    return groups, subgroups


def _compute_grouped_barplot_stats(data, groups, subgroups, agg, compute_std):
    vals = []
    stds = []
    for group_index, group in enumerate(groups):
        group_vals = []
        vals.append(group_vals)
        group_stds = []
        stds.append(group_stds)

        for subgroup_index, subgroup in enumerate(subgroups):
            arr = np.asarray(data.get(group, {}).get(subgroup, []))

            group_vals.append(agg(arr))
            if compute_std:
                group_stds.append(np.std(arr))

    vals = np.asarray(vals)
    stds = np.asarray(stds)

    return vals, stds


def grouped_barplot(
    data,
    agg=None,
    show_std=True,
    cmap_name="tab10",
    ax=None,
    xtick_rotation=30,
):
    if agg is None:
        agg = np.mean

    if ax is None:
        _, ax = plt.subplots()

    groups, subgroups = _get_groups_and_subgroups(data, has_all=True)
    vals, stds = _compute_grouped_barplot_stats(data, groups, subgroups, agg, show_std)

    n_groups = len(groups)
    n_subgroups = len(subgroups)

    x = np.arange(n_groups)
    total_width = 0.8
    bar_width = total_width / max(n_subgroups, 1)
    cmap = plt.get_cmap(cmap_name)
    colors = {subgroups[index]: cmap(index % cmap.N) for index in range(n_subgroups)}

    for subgroup_index, subgroup in enumerate(subgroups):
        offsets = x - total_width / 2 + subgroup_index * bar_width + bar_width / 2

        y = vals[:, subgroup_index]
        yerr = stds[:, subgroup_index] if show_std else None
        ax.bar(
            offsets,
            y,
            width=bar_width,
            yerr=yerr,
            label=subgroup,
            color=colors[subgroup],
            capsize=3,
            ecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=xtick_rotation)

    return ax

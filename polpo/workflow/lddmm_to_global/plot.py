import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from polpo.dataset.plot import get_outer_colors, plot_nested
from polpo.plot.pyplot import plot_hists


def dist_hist(dist_res, ax=None, title="Distribution", **kwargs):
    data = {
        "reconstruction": dist_res.local_reconstruction_error.apply(np.array),
        "local": dist_res.local_pairwise.data,
        "global": dist_res.global_pairwise.data,
    }

    ax = plot_hists(
        data,
        ax=ax,
        density=True,
        histtype="bar",
        edgecolor="black",
        **kwargs,
    )

    ax.set(
        xlabel="Distance",
        title=title,
    )

    return ax


def plot_volume_trends(view, outer_keys=None, ax=None, outer_colors=None):
    if ax is None:
        _, ax = plt.subplots()

    if outer_colors is None:
        outer_colors = get_outer_colors(view.keys())

    one_subject = isinstance(outer_keys, str)
    if outer_keys is None:
        outer_keys = view.dataset.keys()
    elif one_subject:
        outer_keys = [outer_keys]

    dataset = view.dataset.select_outer(outer_keys).map_values(
        lambda x: x.as_pv_surface().volume
    )
    local_meshes = view.local_reconstructed_points.select_outer(outer_keys).map_values(
        lambda x: x.as_pv_surface().volume
    )
    global_meshes = view.global_points.select_outer(outer_keys).map_values(
        lambda x: x.as_pv_surface().volume
    )

    plot_nested(
        dataset,
        ax=ax,
        outer_colors=outer_colors,
        kind="scatter",
        marker="o",
        facecolors="none",
    )

    plot_nested(
        local_meshes,
        ax=ax,
        outer_colors=outer_colors,
        kind="scatter",
        marker="x",
    )

    plot_nested(
        global_meshes,
        ax=ax,
        outer_colors=outer_colors,
        kind="scatter",
        marker="s",
    )

    ax.set_xlabel("Week")
    ax.set_ylabel("Volume")

    condition_markers = {
        "original": "o",
        "local-rec": "x",
        "global": "s",
    }
    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            color="black",
            markerfacecolor="none",
            markersize=8,
            label=condition,
        )
        for condition, marker in condition_markers.items()
    ]

    if not one_subject:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color=color,
                label=outer_id,
                markersize=8,
            )
            for outer_id, color in outer_colors.items()
        ] + handles
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
    )

    return ax


def volume_hist(view, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    volumes = (
        view.dataset.map_values(lambda x: x.as_pv_surface().volume)
        .flatten()
        .values_list()
    )
    volumes_rec = (
        view.local_reconstructed_points.map_values(lambda x: x.as_pv_surface().volume)
        .flatten()
        .values_list()
    )
    volumes_global = (
        view.global_points.map_values(lambda x: x.as_pv_surface().volume)
        .flatten()
        .values_list()
    )

    data = {
        "reconstruction": volumes,
        "local": volumes_rec,
        "global": volumes_global,
    }

    ax = plot_hists(
        data,
        ax=ax,
        density=True,
        histtype="bar",
        edgecolor="black",
        **kwargs,
    )

    ax.set_xlabel("Volume")

    return ax

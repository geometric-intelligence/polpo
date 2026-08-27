import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from polpo.plot.pyplot import plot_grid


def _auto_bins(arrays):
    return np.histogram_bin_edges(
        np.concatenate(arrays),
        bins="auto",
    )


def get_outer_colors(outer_keys, cmap="tab20"):
    outer_keys = list(outer_keys)
    colors = plt.get_cmap(cmap).resampled(len(outer_keys))

    return {key: colors(i) for i, key in enumerate(outer_keys)}


def dist_hist(dist_res, ax=None, title="Distribution"):
    if ax is None:
        fig, ax = plt.subplots()

    rec_local_dists = dist_res.local_reconstruction_error.apply(np.array)
    local_pair_dists = dist_res.local_pairwise.data
    global_pair_dists = dist_res.global_pairwise.data

    arrays = [rec_local_dists, local_pair_dists, global_pair_dists]

    density = True
    hist_type = "bar"

    bins = _auto_bins(arrays)

    for label, array in zip(["reconstruction", "local", "global"], arrays):
        ax.hist(
            array,
            bins=bins,
            edgecolor="black",
            histtype=hist_type,
            density=density,
            label=label,
        )

    ax.set(
        xlabel="Distance",
        ylabel="Density",
        title=title,
    )

    ax.legend()

    return ax


def plot_distance_comparison(
    x_dist,
    y_dist,
    group_by=None,
    colors=None,
    x_label="Local distance",
    y_label="Global distance",
    ax=None,
    identity_line=True,
):
    if ax is None:
        fig, ax = plt.subplots()

    if x_dist.labels != y_dist.labels:
        raise ValueError("Not same key order!")

    x = x_dist.data
    y = y_dist.data

    if group_by is None or colors is None:
        ax.scatter(x, y)
    else:
        categories = [group_by(pair) for pair in x_dist.pairs]
        point_colors = [colors[category] for category in categories]

        ax.scatter(x, y, c=point_colors)

        handles = [
            ax.scatter([], [], color=color, label=label)
            for label, color in colors.items()
        ]

        ax.legend(handles=handles)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if identity_line:
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]

        ax.plot(lims, lims, "--", color="gray")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    return ax


def plot_volume_trends(view, outer_keys=None, ax=None, outer_colors=None):
    if ax is None:
        _, ax = plt.subplots()

    dataset = view.dataset.map_values(lambda x: x.as_pv_surface().volume)
    local_meshes = view.local_reconstructed_points.map_values(
        lambda x: x.as_pv_surface().volume
    )
    global_meshes = view.global_points.map_values(lambda x: x.as_pv_surface().volume)

    one_subject = isinstance(outer_keys, str)
    if outer_keys is None:
        outer_keys = dataset.keys()
    elif one_subject:
        outer_keys = [outer_keys]

    if outer_colors is None:
        outer_colors = get_outer_colors(outer_keys)

    for index, outer_id in enumerate(outer_keys):
        color = outer_colors[outer_id]

        outer_data = dataset.get_outer(outer_id)
        ax.scatter(
            outer_data.keys_list(),
            outer_data.values_list(),
            color=color,
            marker="o",
            facecolors="none",
        )

        outer_data = local_meshes.get_outer(outer_id)
        ax.scatter(
            outer_data.keys_list(),
            outer_data.values_list(),
            marker="x",
            color=color,
        )

        outer_data = global_meshes.get_outer(outer_id)
        ax.scatter(
            outer_data.keys_list(),
            outer_data.values_list(),
            marker="s",
            color=color,
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


def volume_hist(view, ax=None):
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

    arrays = [volumes, volumes_rec, volumes_global]

    density = True
    hist_type = "bar"

    bins = _auto_bins(arrays)

    for label, array in zip(["reconstruction", "local", "global"], arrays):
        ax.hist(
            array,
            bins=bins,
            edgecolor="black",
            histtype=hist_type,
            density=density,
            label=label,
        )

    ax.set(
        xlabel="Volume",
        ylabel="Density",
        title="Distribution",
    )

    ax.legend()

    return ax


def dist_hist_grid(results, **kwargs):
    return plot_grid(results, dist_hist, **kwargs)


def plot_distance_comparison_grid(results, **kwargs):
    def select(item):
        return item.local_pairwise, item.global_pairwise

    return plot_grid(
        results,
        plot_distance_comparison,
        select,
        **kwargs,
    )


def plot_volume_trends_grid(results, **kwargs):
    return plot_grid(results, plot_volume_trends, **kwargs)


def volume_hist_grid(results, **kwargs):
    return plot_grid(results, volume_hist, **kwargs)

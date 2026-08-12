import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

# TODO: TaskRunner?


def _auto_bins(arrays):
    return np.histogram_bin_edges(
        np.concatenate(arrays),
        bins="auto",
    )


def get_subject_colors(subj_ids):
    color_ids = [
        subj_id for subj_id in subj_ids if not str(subj_id).startswith(("3", "4"))
    ]

    tab10 = plt.colormaps["tab10"]
    base_colors = [tab10(i) for i in range(tab10.N) if i != 7]

    cmap = mcolors.ListedColormap(base_colors).resampled(len(color_ids))

    colors = dict(zip(color_ids, cmap(range(len(color_ids)))))
    colors.update(
        {subj_id: "gray" for subj_id in subj_ids if str(subj_id).startswith(("3", "4"))}
    )
    return colors


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

    for label, array in zip(["rec", "local", "global"], arrays):
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


def plot_volume_trends(view, subj_ids=None, ax=None, subj_colors=None):
    if ax is None:
        _, ax = plt.subplots()

    dataset = view.dataset.map_values(lambda x: x.as_pv_surface().volume)
    local_meshes = view.local_reconstructed_points.map_values(
        lambda x: x.as_pv_surface().volume
    )
    global_meshes = view.global_points.map_values(lambda x: x.as_pv_surface().volume)

    one_subject = isinstance(subj_ids, str)
    if subj_ids is None:
        subj_ids = dataset.keys()
    elif one_subject:
        subj_ids = [subj_ids]

    if subj_colors is None:
        subj_colors = get_subject_colors(subj_ids)

    for index, subj_id in enumerate(subj_ids):
        color = subj_colors[subj_id]

        subj_data = dataset.get_outer(subj_id)
        ax.scatter(
            subj_data.keys_list(),
            subj_data.values_list(),
            color=color,
            marker="o",
            facecolors="none",
        )

        subj_data = local_meshes.get_outer(subj_id)
        ax.scatter(
            subj_data.keys_list(),
            subj_data.values_list(),
            marker="x",
            color=color,
        )

        subj_data = global_meshes.get_outer(subj_id)
        ax.scatter(
            subj_data.keys_list(),
            subj_data.values_list(),
            marker="s",
            color=color,
        )

    ax.set_xlabel("Gestational week")
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
                label=subj_id,
                markersize=8,
            )
            for subj_id, color in subj_colors.items()
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

    for label, array in zip(["rec", "local", "global"], arrays):
        ax.hist(
            array,
            bins=bins,
            edgecolor="black",
            histtype=hist_type,
            density=density,
            label=label,
        )

    ax.set(
        xlabel="Volumes",
        ylabel="Density",
        title="Distribution",
    )

    ax.legend()

    return ax

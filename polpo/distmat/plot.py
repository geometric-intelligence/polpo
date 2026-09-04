from matplotlib import pyplot as plt


def plot_dist_mat(dists, title=None, fig_size=None):
    # TODO: add as_method to PairwiseDists
    fig, ax = plt.subplots(figsize=fig_size)

    im = ax.imshow(dists.matrix)

    plt.colorbar(im)

    if title is not None:
        ax.set_title(title)

    keys = dists.labels
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=90)

    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys)

    return ax


def _group_pairs(pairs, grouper):
    groups = {}

    for pair in pairs:
        group = grouper(*pair)
        groups.setdefault(group, []).append(pair)

    return groups


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
        groups = _group_pairs(x_dist.pairs, group_by)

        for category, pairs in groups.items():
            x_group = x_dist.select_pairs(pairs)
            y_group = y_dist.select_pairs(pairs)

            ax.scatter(
                x_group.data,
                y_group.data,
                color=colors[category],
                label=category,
            )

        ax.legend()

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

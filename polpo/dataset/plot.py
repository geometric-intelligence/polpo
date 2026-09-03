from matplotlib import pyplot as plt


def get_outer_colors(outer_keys, cmap="tab20"):
    outer_keys = list(outer_keys)
    colors = plt.get_cmap(cmap).resampled(len(outer_keys))

    return {key: colors(i) for i, key in enumerate(outer_keys)}


def plot_nested(
    data,
    ax=None,
    outer_colors=None,
    include_label=False,
    kind="plot",
    **kwargs,
):
    if ax is None:
        _, ax = plt.subplots()

    if outer_colors is None:
        outer_colors = get_outer_colors(data.keys())

    plot = getattr(ax, kind)

    for outer_key, outer_data in data.iter_outer():
        plot(
            outer_data.keys_list(),
            outer_data.values_list(),
            color=outer_colors[outer_key],
            label=outer_key if include_label else None,
            **kwargs,
        )

    if include_label:
        ax.legend()

    return ax

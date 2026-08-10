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

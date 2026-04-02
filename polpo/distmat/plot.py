import numpy as np
from matplotlib import pyplot as plt

from ._distmat import sort_dist_mat


def plot_dist_mat(dists, keys, title=None, fig_size=None, sort=False):
    if sort:
        dists, sorted_idx = sort_dist_mat(dists)
        keys = np.asarray(keys)[sorted_idx]

    fig, ax = plt.subplots(figsize=fig_size)

    im = ax.imshow(dists)

    plt.colorbar(im)

    if title is not None:
        ax.set_title(title)

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=90)

    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys)

    return ax

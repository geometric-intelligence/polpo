import numpy as np
from matplotlib import pyplot as plt


def plot_slices(slices, cmap="gray", vertical=False, ax=None, **imshow_kwargs):
    if isinstance(slices, np.ndarray):
        n_slices = 1
        slices = [slices]
    else:
        n_slices = len(slices)

    n_rows, n_cols = 1, n_slices
    if vertical:
        n_rows, n_cols = n_cols, n_rows

    axes = ax
    if ax is None:
        _, axes = plt.subplots(n_rows, n_cols, constrained_layout=True)

    if n_slices == 1:
        axes = [axes]

    for ax, slice_ in zip(axes, slices):
        ax.imshow(
            slice_.T,
            cmap=cmap,
            origin="lower",
            **imshow_kwargs,
        )

    return axes

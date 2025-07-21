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

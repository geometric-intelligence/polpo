import numpy as np
from matplotlib import pyplot as plt

from .base import Plotter


class SlicesPlotter(Plotter):
    def __init__(self, cmap="gray"):
        super().__init__()
        self.cmap = cmap

    def plot(self, slices):
        if isinstance(slices, np.ndarray):
            n_slices = 1
            slices = [slices]
        else:
            n_slices = len(slices)

        fig, axes = plt.subplots(1, n_slices, constrained_layout=True)
        if n_slices == 1:
            axes = [axes]

        for ax, slice_ in zip(axes, slices):
            ax.imshow(
                slice_.T,
                cmap=self.cmap,
                origin="lower",
            )

        return fig, axes

import matplotlib.pyplot as plt
import numpy as np

from .base import Plotter

import plotly.graph_objects as go
from plotly.subplots import make_subplots

class SlicesPlotter(Plotter):
    def __init__(self, cmap="gray"):
        self.cmap = cmap

    def plot(self, slices):
        if isinstance(slices, np.ndarray):
            slices = [slices]

        n_slices = len(slices)
        fig = make_subplots(rows=1, cols=n_slices, horizontal_spacing=0)

        for i, slice_ in enumerate(slices):
            fig.add_trace(
                go.Image(z=slice_.T),
                row=1,
                col=i + 1,
            )
            fig.update_xaxes(showticklabels=False, row=1, col=i + 1)
            fig.update_yaxes(showticklabels=False, scaleanchor=f"x{i+1}", row=1, col=i + 1)

        fig.update_layout(
            autosize=True,
            height=None,
            width=None,
            margin=dict(t=20, b=20, l=0, r=0),
        )

        return fig

# class SlicesPlotter(Plotter):
#     def __init__(self, cmap="gray"):
#         self.cmap = cmap

#     def plot(self, slices):
#         if isinstance(slices, np.ndarray):
#             n_slices = 1
#             slices = [slices]
#         else:
#             n_slices = len(slices)

#         fig, axes = plt.subplots(1, n_slices, constrained_layout=True)
#         if n_slices == 1:
#             axes = [axes]

#         for ax, slice_ in zip(axes, slices):
#             ax.imshow(slice_.T, cmap=self.cmap, origin="lower",)

#         return fig, axes

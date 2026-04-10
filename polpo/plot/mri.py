"""Package agnostic imports for MRI plotters."""

try:
    from .pyplot import SlicesPlotter  # noqa:F401
except ImportError:
    pass

try:
    from .plotly import SlicePlotter  # noqa:F401
except ImportError:
    pass

from polpo.volume.slicing import VolumeSlicer as MriSlicer

from .base import Plotter


class MriPlotter(Plotter):
    def __init__(self, plotter=None, slicer=None, label_axes=True, add_title=True):
        if slicer is None:
            slicer = MriSlicer()

        if plotter is None:
            plotter = SlicesPlotter()

        self.slicer = slicer
        self.plotter = plotter
        self.label_axes = label_axes
        self.add_title = add_title

        self._default_labels = {0: ("Y", "Z"), 1: ("X", "Z"), 2: ("X", "Y")}
        self._default_titles = {0: "Side View", 1: "Front View", 2: " Top View"}

    @classmethod
    def build(
        cls,
        *args,
        slicer=None,
        index_ordering=(0, 1, 2),
        common_size=None,
        plotter=None,
        cmap="gray",
        **kwargs,
    ):
        # convenient instantiation
        if slicer is None:
            slicer = MriSlicer(index_ordering, common_size)

        if plotter is None:
            plotter = SlicesPlotter(cmap)

        return cls(*args, slicer=slicer, plotter=plotter, **kwargs)

    def plot(self, img_fdata, slice_indices):
        fig, axes = self.plotter.plot(self.slicer.take_slices(img_fdata, slice_indices))

        if self.label_axes:
            for ax, index in zip(axes, self.slicer.index_ordering):
                x_label, y_label = self._default_labels[index]
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)

        if self.add_title:
            for ax, index in zip(axes, self.slicer.index_ordering):
                ax.set_title(self._default_titles[index])

        return fig, axes

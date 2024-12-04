import abc

import numpy as np
import plotly.graph_objects as go


def return_nii_plot(
    raw_mri_datum,
    x,
    y,
    z,
):  # week,
    """Return the nii plot based on the week and the x, y, z coordinates."""
    slice_0 = raw_mri_datum[x, :, :]
    slice_1 = raw_mri_datum[:, y, :]
    slice_2 = raw_mri_datum[:, :, z]

    # TODO: can be much simplified

    common_width = max(len(slice_0[:, 0]), len(slice_1[:, 0]), len(slice_2[:, 0]))
    common_height = max(len(slice_0[0]), len(slice_1[0]), len(slice_2[0]))

    slices = [slice_0, slice_1, slice_2]
    for i_slice, slice in enumerate([slice_0, slice_1, slice_2]):
        if len(slice[:, 0]) < common_width:
            diff = common_width - len(slice[:, 0])
            # slice = np.pad(slice, ((0, diff), (0, 0)), mode="constant")
            slice = np.pad(slice, ((diff // 2, diff // 2), (0, 0)), mode="constant")
            slices[i_slice] = slice
        if len(slice[0]) < common_height:
            diff = common_height - len(slice[0])
            # slice = np.pad(slice, ((0, 0), (0, diff)), mode="constant")
            slice = np.pad(slice, ((0, 0), (diff // 2, diff // 2)), mode="constant")
            slices[i_slice] = slice

    side_fig = plot_slice_as_plotly(
        slices[0], cmap="gray", title="Side View", x_label="Y", y_label="Z"
    )
    front_fig = plot_slice_as_plotly(
        slices[1], cmap="gray", title="Front View", x_label="X", y_label="Z"
    )
    top_fig = plot_slice_as_plotly(
        slices[2], cmap="gray", title="Top View", x_label="X", y_label="Y"
    )

    return side_fig, front_fig, top_fig


def plot_slice_as_plotly(
    one_slice, cmap="gray", title="Slice Visualization", x_label="X", y_label="Y"
):
    """Display an image slice as a Plotly figure."""
    # Create heatmap trace for the current slice
    heatmap_trace = go.Heatmap(z=one_slice.T, colorscale=cmap, showscale=False)

    width = int(len(one_slice[:, 0]) * 1.5)
    height = int(len(one_slice[0]) * 1.5)

    layout = go.Layout(
        title=title,
        title_x=0.5,
        xaxis=dict(title=x_label),
        yaxis=dict(title=y_label),
        width=width,
        height=height,
    )

    # Create a Plotly figure with the heatmap trace
    fig = go.Figure(data=heatmap_trace, layout=layout)

    # Update layout to adjust appearance
    # fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)

    return fig


class Plotter(abc.ABC):
    @abc.abstractmethod
    def plot(self, data):
        pass


class SlicePlotter(Plotter):
    # TODO: can be further improved
    # TODO: explore plotly.graph_objects and check if it should be a component

    def __init__(
        self, cmap="gray", title="Slice Visualization", x_label="X", y_label="Y"
    ):
        self.cmap = cmap
        self.title = title
        self.x_label = x_label
        self.y_label = y_label

    def plot(self, data):
        return plot_slice_as_plotly(
            data,
            cmap=self.cmap,
            title=self.title,
            x_label=self.x_label,
            y_label=self.y_label,
        )


class MeshPlotter(Plotter):
    # TODO: can be further improved
    def __init__(self):
        self.layout = go.Layout(
            margin=go.layout.Margin(
                l=0,
                r=0,
                b=0,
                t=0,
            ),
            width=700,
            height=700,
            scene=dict(
                aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"
            ),
        )

    def plot(self, mesh):
        # TODO: need to get access to previous image for nicer transition?

        # TODO: add version that stores mesh connectivities, so only the vertices are passed?
        mesh_pred = mesh.vertices
        faces = mesh.faces
        vertex_colors = mesh.visual.vertex_colors

        return go.Figure(
            data=[
                go.Mesh3d(
                    x=mesh_pred[:, 0],
                    y=mesh_pred[:, 1],
                    z=mesh_pred[:, 2],
                    colorbar_title="z",
                    vertexcolor=vertex_colors,
                    # i, j and k give the vertices of triangles
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    name="y",
                )
            ],
            layout=self.layout,
        )

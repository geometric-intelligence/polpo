import abc

import plotly.graph_objects as go

from .base import Plotter


class GoPlotter(Plotter, abc.ABC):
    @abc.abstractmethod
    def transform_data(self, data):
        pass

    def update(self, fig, data):
        fig.update(data=self.transform_data(data))
        return fig


class SlicePlotter(GoPlotter):
    def __init__(
        self, cmap="gray", title="Slice Visualization", x_label="X", y_label="Y"
    ):
        self.cmap = cmap
        self.title = title
        self.x_label = x_label
        self.y_label = y_label

        self.layout = go.Layout(
            title=self.title,
            title_x=0.5,
            xaxis=dict(title=self.x_label),
            yaxis=dict(title=self.y_label),
            uirevision="constant",
        )

    def transform_data(self, data):
        return [go.Heatmap(z=data.T, colorscale=self.cmap, showscale=False)]

    def plot(self, data=None):
        # Create heatmap trace for the current slice
        if data is None:
            return go.Figure(
                layout=self.layout,
            )

        fig = go.Figure(data=self.update(data), layout=self.layout)

        width = int(len(data[:, 0]) * 1.5)
        height = int(len(data[0]) * 1.5)

        fig.update_layout(
            width=width,
            height=height,
        )

        return fig


class MeshPlotter(GoPlotter):
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
            uirevision="constant",
        )

    def transform_data(self, mesh):
        mesh_pred = mesh.vertices
        faces = mesh.faces
        vertex_colors = mesh.visual.vertex_colors

        data = [
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
        ]
        return data

    def plot(self, mesh=None):
        if mesh is None:
            return go.Figure(
                layout=self.layout,
            )
        return go.Figure(
            data=self.transform_data(mesh),
            layout=self.layout,
        )

import abc

import plotly.graph_objects as go
import numpy as np

from .base import Plotter


class GoPlotter(Plotter, abc.ABC):
    @abc.abstractmethod
    def transform_data(self, data):
        pass

    def update(self, fig, data):
        fig.update(data=self.transform_data(data))
        return fig

    def plot(self, data):
        fig = go.Figure(data=self.update(data), layout=self.layout)
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

        fig = go.Figure(data=self.transform_data(data), layout=self.layout)

        width = int(len(data[:, 0]) * 1.5)
        height = int(len(data[0]) * 1.5)

        fig.update_layout(
            width=width,
            height=height,
        )

        return fig

class MeshPlotter:
    def __init__(self, bounds=None, bounds_with_template=None, template_mesh=None):
        self.layout = go.Layout(
            margin=go.layout.Margin(l=0, r=0, b=0, t=0),
            width=700,
            height=700,
            scene=dict(
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True),
                zaxis=dict(showgrid=True),
                aspectmode="manual",
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="z",
                aspectratio=dict(x=1, y=1, z=1),
            ),
            uirevision="constant",
        )
        self.bounds = bounds
        self.bounds_with_template = bounds_with_template
        self.template_mesh = template_mesh["mesh"]

    def transform_data(self, mesh, template=None, show_template=False):
        """Convert multiple meshes into Plotly-compatible data."""
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
            ),  
        ]

        if show_template:
            mesh_template = self.template_mesh.vertices
            faces_template = self.template_mesh.faces
            vertex_colors_template = np.full((len(mesh_template), 3), [0.678, 0.847, 0.902])
            data.append(
                go.Mesh3d(
                    x=mesh_template[:, 0],
                    y=mesh_template[:, 1],
                    z=mesh_template[:, 2],
                    colorbar_title="z",
                    vertexcolor=vertex_colors_template,
                    i=faces_template[:, 0],
                    j=faces_template[:, 1],
                    k=faces_template[:, 2],
                    opacity=0.5,
                    visible=show_template,  # Initially set the overlay to be invisible
                    flatshading=False,
                    lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.5)
                )
            )
  
        return data

    # def plot(self, meshes):
    #     """Update the figure dynamically."""
    #     mins, maxs = self.compute_bounding_box(meshes)
    #     self.layout.scene.xaxis.range = [mins[0], maxs[0]]
    #     self.layout.scene.yaxis.range = [mins[1], maxs[1]]
    #     self.layout.scene.zaxis.range = [mins[2], maxs[2]]

    #     return go.Figure(data=self.transform_data(mesh), layout=self.layout)

    def plot(self, mesh=None, template=None, show_template=False):
        if mesh is None:
            return go.Figure(
                layout=self.layout,
            )

        if show_template:
            mins, maxs = self.bounds_with_template
        else:
            mins, maxs = self.bounds

        self.layout.scene.xaxis.range = [mins[0], maxs[0]]
        self.layout.scene.yaxis.range = [mins[1], maxs[1]]
        self.layout.scene.zaxis.range = [mins[2], maxs[2]]
        return go.Figure(
            data=self.transform_data(mesh, template, show_template),
            layout=self.layout,
        )


# class MeshPlotter(GoPlotter):
#     def __init__(self, meshes):
#         # Compute the bounding box union
#         self.update_bounding_box(meshes)

#     def update_bounding_box(self, meshes):
#         """
#         Updates the bounding box to enclose all given meshes.
#         """
#         if not meshes:
#             # Default bounds if no meshes are provided
#             self.x_min, self.y_min, self.z_min = -1, -1, -1
#             self.x_max, self.y_max, self.z_max = 1, 1, 1
#         else:
#             # Collect all bounding boxes
#             all_bounds = [mesh.bounding_box.bounds for mesh in meshes]

#             # Stack and compute global min/max
#             mins = np.min([b[0] for b in all_bounds], axis=0)
#             maxs = np.max([b[1] for b in all_bounds], axis=0)

#             self.x_min, self.y_min, self.z_min = mins
#             self.x_max, self.y_max, self.z_max = maxs

#         # Update the layout with new bounds
#         self.layout = go.Layout(
#             margin=go.layout.Margin(l=0, r=0, b=0, t=0),
#             width=700,
#             height=700,
#             scene=dict(
#                 xaxis=dict(range=[self.x_min, self.x_max], showgrid=True),
#                 yaxis=dict(range=[self.y_min, self.y_max], showgrid=True),
#                 zaxis=dict(range=[self.z_min, self.z_max], showgrid=True),
#                 aspectmode="manual",
#                 xaxis_title="x",
#                 yaxis_title="y",
#                 zaxis_title="z",
#                 aspectratio=dict(x=1, y=1, z=1),
#             ),
#             uirevision="constant",
#         )

#     def transform_data(self, mesh):
#         mesh_pred = mesh.vertices
#         faces = mesh.faces
#         vertex_colors = mesh.visual.vertex_colors

#         data = [
#             go.Mesh3d(
#                 x=mesh_pred[:, 0],
#                 y=mesh_pred[:, 1],
#                 z=mesh_pred[:, 2],
#                 colorbar_title="z",
#                 vertexcolor=vertex_colors,
#                 # i, j and k give the vertices of triangles
#                 i=faces[:, 0],
#                 j=faces[:, 1],
#                 k=faces[:, 2],
#                 name="y",
#             )
#         ]
#         return data

#     def plot(self, mesh=None):
#         if mesh is None:
#             return go.Figure(
#                 layout=self.layout,
#             )
#         return go.Figure(
#             data=self.transform_data(mesh),
#             layout=self.layout,
#         )

#     def update_layout(self, fig, mesh):
#         # Update the layout with the new mesh data
#         fig.update(data=self.transform_data(mesh))
#         return fig

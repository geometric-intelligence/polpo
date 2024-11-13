"""Components."""

import abc

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from .callbacks import (
    create_show_mesh,
    create_view_model_update,
)
from .models import MriSlices, PdDfLookupModel
from .plot import SlicePlotter
from .style import STYLE as S
from .utils import unnest_list


class Component(abc.ABC):
    def __init__(self, id_prefix=""):
        self.id_prefix = id_prefix

    @abc.abstractmethod
    def to_dash(self):
        # NB: expects list[dash.Component] as output
        pass


class VarDefComponent(Component, abc.ABC):
    def __init__(self, var_def, id_prefix="", id_suffix=""):
        super().__init__(id_prefix)
        self.var_def = var_def
        self.id_suffix = id_suffix

    @property
    def id(self):
        return f"{self.id_prefix}{self.var_def.id}{self.id_suffix}"


class IdComponent(Component, abc.ABC):
    # TODO: improve these abstractions
    def __init__(self, id_, id_prefix="", id_suffix=""):
        super().__init__(id_prefix)
        self.id_ = id_
        self.id_suffix = id_suffix

    @property
    def id(self):
        return f"{self.id_prefix}{self.id_}{self.id_suffix}"


class BaseComponentGroup(Component, abc.ABC):
    def __init__(self, components, id_prefix=""):
        self.components = components
        super().__init__(id_prefix)

    @property
    def id_prefix(self):
        return self._id_prefix

    @id_prefix.setter
    def id_prefix(self, value):
        self._id_prefix = value
        if value:
            for component in self.components:
                component.id_prefix = value


class ComponentGroup(BaseComponentGroup):
    def __init__(self, components, id_prefix="", title=None):
        super().__init__(components, id_prefix)
        self.title = title

    def __getitem__(self, index):
        return self.components[index]

    def to_dash(self, data=None):
        if data is not None:
            return unnest_list(
                [component.to_dash(value) for component, value in zip(self, data)]
            )

        title_label = (
            [
                dbc.Label(
                    f"{self.title}:",
                    style={
                        "font-size": S.text_fontsize,
                        "fontFamily": S.text_fontfamily,
                    },
                ),
            ]
            if self.title
            else []
        )

        return title_label + unnest_list([component.to_dash() for component in self])

    def as_output(self):
        return unnest_list(component.as_output() for component in self)

    def as_empty_output(self):
        return unnest_list(component.as_empty_output() for component in self)

    def as_input(self):
        return unnest_list(component.as_input() for component in self)


class Slider(VarDefComponent):
    """A slider."""

    def __init__(self, var_def, step=1, id_prefix="", label_style=None):
        super().__init__(var_def=var_def, id_prefix=id_prefix, id_suffix="-slider")
        # TODO: think more about this design

        self.step = step

        # TODO: can default be set for the general app instead?
        default_label_style = {
            "fontSize": S.text_fontsize,
            "fontFamily": S.text_fontfamily,
        }
        self.label_style = (label_style or {}).update(default_label_style)

    def __repr__(self):
        return f"Slider({self.id})"

    def to_dash(self):
        # TODO: allow to config from config file, e.g. label_style
        label = dbc.Label(
            self.var_def.label,
            style=self.label_style,
        )
        slider = dcc.Slider(
            id=self.id,
            min=self.var_def.min_value,
            max=self.var_def.max_value,
            step=self.step,
            value=self.var_def.default_value,
            marks={
                self.var_def.min_value: {"label": "min"},
                self.var_def.max_value: {"label": "max"},
            },
            tooltip={
                "placement": "bottom",
                "always_visible": True,
                "style": {"fontSize": "30px", "fontFamily": S.text_fontfamily},
            },
        )

        return [label, slider]

    def as_input(self):
        return [Input(self.id, "drag_value")]


class DepVar(VarDefComponent):
    def __repr__(self):
        return f"DepVar({self.var_def.id})"

    def to_dash(self, value=None):
        if value:
            return [f"{self.var_def.label}: {value}"]

        return [
            html.Div(
                id=self.id,
                style={
                    "font-size": S.text_fontsize,
                    "fontFamily": S.text_fontfamily,
                },
            )
        ]

    def as_output(self):
        return [Output(self.id, "children")]

    def as_empty_output(self):
        return [""]


class Graph(IdComponent):
    # TODO: any links with dash-vtk abstractions?

    def __init__(self, id_, plotter=None, id_prefix="", id_suffix=""):
        # TODO: add reasonable default plotter
        super().__init__(id_, id_prefix, id_suffix)
        self.plotter = plotter

    def to_dash(self, data=None):
        if data is not None:
            return [self.plotter.plot(data)]

        return [
            dcc.Graph(
                id=self.id,
                config={"displayModeBar": False},
            )
        ]

    def as_output(self):
        return [Output(self.id, "figure")]

    def as_empty_output(self):
        return [go.Figure()]


class GraphRow(ComponentGroup):
    def __init__(self, n_graphs=3, graphs=None, id_prefix=""):
        # NB: `n_graphs`` is ignored if `graphs` is not None

        if graphs is None:
            graphs = [
                Graph(id_="plot", id_suffix=f"-{index}") for index in range(n_graphs)
            ]

        super().__init__(components=graphs, id_prefix=id_prefix)

    def to_dash(self, data=None):
        if data is not None:
            return super().to_dash(data)

        return [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            graph.to_dash(),
                            style={"paddingTop": "0px"},
                        ),
                        sm=4,
                    )
                    for graph in self
                ],
                align="center",
                style={
                    "marginLeft": S.margin_side,
                    "marginRight": S.margin_side,
                    "marginTop": "50px",
                },
            )
        ]


class MriSliders(ComponentGroup):
    def __init__(self, components, trims=((20, 40), 50, 70), id_prefix="", title=None):
        super().__init__(components, id_prefix=id_prefix, title=title)
        self.trims = [(trim, trim) if isinstance(trim, int) else trim for trim in trims]

    def update_lims(self, mri_data):
        self[0].var_def.default_value = self[0].var_def.default_value or 1
        self[0].var_def.max_value = min(self[0].var_def.max_value, len(mri_data))
        for slider, trim in zip(self[1:], self.trims):
            var_def = slider.var_def
            min_value = var_def.min_value or trim[0]
            max_value = var_def.max_value or (mri_data[0].shape[0] - 1 - trim[1])
            var_def.default_value = (
                var_def.default_value or (max_value - min_value) // 2 + min_value
            )
            var_def.min_value = min_value
            var_def.max_value = max_value


class MriGraphRow(GraphRow):
    # NB: just syntax sugar

    def __init__(self):
        titles = ("Side View", "Front View", "Top View")
        x_labels = ("Y", "X", "X")
        y_labels = ("Z", "Z", "Y")
        graphs = [
            Graph(
                id_="plot",
                id_suffix=f"-{index}",
                plotter=SlicePlotter(title=title, x_label=x_label, y_label=y_label),
            )
            for index, (title, x_label, y_label) in enumerate(
                zip(titles, x_labels, y_labels)
            )
        ]
        super().__init__(id_prefix="nii-", graphs=graphs)


class MriExplorer(BaseComponentGroup):
    # data
    # plots
    # sliders
    # session info card

    # also instructions?

    # TODO: check if multiple callbacks can be defined
    def __init__(
        self,
        mri_data,
        hormones_df,
        sliders,
        session_info,
        graph_row=None,
        id_prefix="",
    ):
        if graph_row is None:
            graph_row = MriGraphRow()

        # TODO: used to train the model and to update the controller
        self.mri_data = mri_data
        self.hormones_df = hormones_df

        # NB: an input view
        self.sliders = sliders
        # NB: an output view of the brain data
        self.graph_row = graph_row
        # NB: a model of the brain data
        self.mri_model = MriSlices(self.mri_data)

        # NB: a view of the hormones data
        self.session_info = session_info
        # NB: a model of the hormones data
        self.session_info_model = PdDfLookupModel(
            df=hormones_df,
            output_keys=[elem.var_def.id for elem in session_info],
            tar=1,
        )

        super().__init__([sliders, graph_row, session_info], id_prefix)

    def _create_callbacks(self):
        create_view_model_update(self.sliders, self.graph_row, self.mri_model)
        create_view_model_update(
            self.sliders[0], self.session_info, self.session_info_model
        )

    def to_dash(self):
        if hasattr(self.slides, "update_lims"):
            self.sliders.update_lims(self.mri_data)

        # TODO: allow it as input?
        instructions_text = dbc.Row(
            [
                html.P(
                    [
                        "Use the 'Session Number' slider to flip through T1 brain data from each MRI session. Use the X, Y, Z coordinate sliders choose the MRI slice. Additional information about the session will be displayed to the right of the sliders.",
                    ],
                    style={
                        "fontSize": S.text_fontsize,
                        "fontFamily": S.text_fontfamily,
                        "marginLeft": S.margin_side,
                        "marginRight": S.margin_side,
                    },
                ),
            ],
        )

        plots_card = self.graph_row.to_dash()
        plots = dbc.Row(
            [
                dbc.Col(plots_card, sm=14),
            ],
            align="center",
            style={
                "marginLeft": S.margin_side,
                "marginRight": S.margin_side,
                "marginTop": "50px",
            },
        )

        sliders_card = dbc.Card(
            [
                dbc.Stack(
                    self.sliders.to_dash(),
                    gap=3,
                )
            ],
            body=True,
        )
        sliders_column = [
            dbc.Row(sliders_card),
        ]

        session_info = self.session_info.to_dash()
        sess_info_card = dbc.Card(
            [
                dbc.Stack(
                    session_info,
                    gap=3,
                )
            ],
            body=True,
        )

        sliders_and_session = dbc.Row(
            [
                dbc.Col(sliders_column, sm=7, width=700),
                dbc.Col(sess_info_card, sm=4, width=700),
            ],
            align="center",
            style={
                "marginLeft": S.margin_side,
                "marginRight": S.margin_side,
                "marginTop": "50px",
            },
        )

        self._create_callbacks()

        return [instructions_text, plots, sliders_and_session]


class MeshGraph(IdComponent):
    # TODO: give it a suffix of plot?

    def to_dash(self, mesh=None):
        # TODO: accept figure/mesh for update? may need to rename if the case
        # TODO: this handles meshes for now
        if mesh is None:
            return [
                dcc.Graph(
                    id=self.id,
                )
            ]

        # TODO: define layout at startup
        layout = go.Layout(
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

        # TODO: need to get access to previous image for nicer transition

        # TODO: create an update() method instead of calling this again?

        # TODO: update
        mesh_pred = mesh.vertices
        faces = mesh.faces

        fig = go.Figure(
            data=[
                go.Mesh3d(
                    x=mesh_pred[:, 0],
                    y=mesh_pred[:, 1],
                    z=mesh_pred[:, 2],
                    colorbar_title="z",
                    # vertexcolor=vertex_colors, # TODO: uncomment
                    # i, j and k give the vertices of triangles
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    name="y",
                )
            ],
            layout=layout,
        )
        return fig


class MeshExplorer(BaseComponentGroup):
    def __init__(self, mesh_data, graph, sliders, id_prefix=""):
        # TODO: adapt sliders to get toggle button

        # TODO: controller instead of sliders? apply same in MRI explorer
        # TODO: allow also for controlled? like in MRI explorer

        self.mesh_data = mesh_data
        self.graph = graph
        self.sliders = sliders

    def to_dash(self):
        # TODO: update

        # TODO: add callback

        graph = self.graph.to_dash()
        sliders = self.sliders.to_dash()

        create_show_mesh(self.mesh_data, self.graph, self.sliders)

        return [graph, sliders]

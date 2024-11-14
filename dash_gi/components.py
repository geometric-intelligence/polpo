"""Components."""

import abc

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from .callbacks import (
    create_button_toggler_for_view_model_update,
    create_view_model_update,
)
from .models import (
    ConstantOutputModel,
    LinearMeshVertexScaling,
    MriSlices,
    PdDfLookupModel,
)
from .plot import SlicePlotter
from .style import STYLE as S
from .utils import unnest_list


class Component(abc.ABC):
    def __init__(self, id_prefix=""):
        self.id_prefix = id_prefix

    @abc.abstractmethod
    def to_dash(self):
        # NB: returns list[dash.Component]
        pass

    def as_output(self, component_property, allow_duplicate=False):
        return [Output(self.id, component_property, allow_duplicate=allow_duplicate)]

    def prefix(self, name):
        return f"{self.id_prefix}{name}"


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

    def __getitem__(self, index):
        return self.components[index]

    def __len__(self):
        return len(self.components)

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

    def as_output(self, component_property=None, allow_duplicate=False):
        return unnest_list(
            component.as_output(
                component_property=component_property, allow_duplicate=allow_duplicate
            )
            for component in self
        )

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

    def as_output(self, component_property=None, allow_duplicate=False):
        component_property = component_property or "children"
        return [Output(self.id, component_property, allow_duplicate=allow_duplicate)]

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

    def as_output(self, component_property=None, allow_duplicate=False):
        component_property = component_property or "figure"
        return [Output(self.id, component_property, allow_duplicate=allow_duplicate)]

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
        if hasattr(self.sliders, "update_lims"):
            self.sliders.update_lims(self.mri_data)

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


class MeshExplorer(BaseComponentGroup):
    def __init__(
        self,
        mesh_data,
        graph,
        hormone_sliders,
        week_sliders,
        id_prefix="",
    ):
        # TODO: disallow ids and control it from here
        self.graph = graph
        self.hormone_sliders = hormone_sliders
        self.week_sliders = week_sliders

        # TODO: an experiment, just for debugging
        self.hormone_mesh_model = ConstantOutputModel(mesh_data)
        self.week_mesh_model = LinearMeshVertexScaling(mesh_data)

        super().__init__(
            [self.graph, self.hormone_sliders, self.week_sliders], id_prefix=id_prefix
        )

        self._toggler_button_id = "button"
        self._week_sliders_card_id = "gest_week_slider_container"
        self._hormone_sliders_card_id = "hormone_slider_container"

    def to_dash(self):
        instructions_text = dbc.Row(
            [
                html.P(
                    [
                        "Use the hormone sliders or the gestational week slider to adjust observe the predicted shape changes in the left hippocampal formation.",
                        html.Br(),
                    ],
                    style={
                        "fontSize": S.text_fontsize,
                        "fontFamily": S.text_fontfamily,
                    },
                ),
            ],
        )

        graph = self.graph.to_dash()
        week_sliders = self.week_sliders.to_dash()
        hormone_sliders = self.hormone_sliders.to_dash()

        week_slider_card = HideableComponent(
            id_=self._week_sliders_card_id,
            dash_component=dbc.Card(
                dbc.Stack(
                    week_sliders,
                    gap=3,
                ),
                body=True,
            ),
            id_prefix=self.id_prefix,
        )

        hormone_sliders_card = HideableComponent(
            id_=self._hormone_sliders_card_id,
            dash_component=dbc.Card(
                dbc.Stack(
                    hormone_sliders,
                    gap=3,
                ),
                body=True,
            ),
            id_prefix=self.id_prefix,
        )

        toggle_id = self.prefix(self._toggler_button_id)
        sliders_column = [
            html.Button(
                "Click Here to Toggle Between Gestational Week vs Hormone Value Prediction",
                id=toggle_id,
                n_clicks=0,
            ),
            dbc.Row(week_slider_card.to_dash()),
            dbc.Row(hormone_sliders_card.to_dash()),
        ]

        out = [
            instructions_text,
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            graph,
                            style={"paddingTop": "0px"},
                        ),
                        sm=4,
                        width=700,
                    ),
                    dbc.Col(sm=3, width=100),
                    dbc.Col(sliders_column, sm=4, width=700),
                ],
                align="center",
                style={
                    "marginLeft": S.margin_side,
                    "marginRight": S.margin_side,
                    "marginTop": "50px",
                },
            ),
        ]

        create_button_toggler_for_view_model_update(
            output_view=self.graph,
            input_views=[self.week_sliders, self.hormone_sliders],
            models=[self.week_mesh_model, self.hormone_mesh_model],
            toggle_id=toggle_id,
            hideable_components=[week_slider_card, hormone_sliders_card],
        )

        return out


class HideableComponent(IdComponent):
    def __init__(self, id_, dash_component, id_prefix="", id_suffix=""):
        super().__init__(id_, id_prefix, id_suffix)
        self.dash_component = dash_component

    def to_dash(self):
        return html.Div(
            children=[self.dash_component],
            id=self.id,
        )

"""Components."""

import abc

import dash_bootstrap_components as dbc
from dash import dcc, html

from .callbacks import create_update_nii_plot_basic, create_update_session_info
from .style import STYLE as S
from .utils import unnest_list


class Component(abc.ABC):
    def __init__(self, prefix_id=""):
        self.prefix_id = prefix_id

    @abc.abstractmethod
    def to_dash(self):
        # NB: expects list[dash.Component] as output
        pass


class BaseComponentGroup(Component, abc.ABC):
    def __init__(self, components, prefix_id=""):
        self.components = components
        super().__init__(prefix_id)

    @property
    def prefix_id(self):
        return self._prefix_id

    @prefix_id.setter
    def prefix_id(self, value):
        self._prefix_id = value
        if value:
            for component in self.components:
                component.prefix_id = value


class ComponentGroup(BaseComponentGroup):
    def __init__(self, components, prefix_id="", title=None):
        super().__init__(components, prefix_id)
        self.title = title

    def __getitem__(self, index):
        return self.components[index]

    def to_dash(self):
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

        return title_label + unnest_list(
            [component.to_dash() for component in self.components]
        )


class Slider(Component):
    """A slider."""

    def __init__(self, var_def, step=1, prefix_id="", label_style=None):
        super().__init__(prefix_id)
        # TODO: think more about this design
        # TODO: add a prefix to the id

        self.var_def = var_def
        self.step = step

        # TODO: can default be set for the general app instead?
        default_label_style = {
            "fontSize": S.text_fontsize,
            "fontFamily": S.text_fontfamily,
        }
        self.label_style = (label_style or {}).update(default_label_style)

    def __repr__(self):
        return f"Slider({self.var_def.id})"

    @property
    def id(self):
        return f"{self.prefix_id}{self.var_def.id}-slider"

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


class DepVar(Component):
    def __init__(self, var_def, prefix_id=""):
        super().__init__(prefix_id)
        self.var_def = var_def

    def __repr__(self):
        return f"DepVar({self.var_def.id})"

    @property
    def id(self):
        return f"{self.prefix_id}{self.var_def.id}"

    def to_dash(self, value=None):
        if value:
            return f"self.var_def.label: {value}"

        return [
            html.Div(
                id=self.id,
                style={
                    "font-size": S.text_fontsize,
                    "fontFamily": S.text_fontfamily,
                },
            )
        ]


class GraphRow(Component):
    def __init__(self, id_="nii", n_graphs=3, prefix_id=""):
        super().__init__(prefix_id)
        self.id_ = id_
        self.n_graphs = n_graphs

    @property
    def id(self):
        return f"{self.prefix_id}{self.id_}-plot"

    def get_id(self, index):
        return f"{self.id}-{index}"

    def ids(self):
        return [self.get_id(index) for index in range(self.n_graphs)]

    def to_dash(self):
        return [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            dcc.Graph(
                                id=self.get_id(index),
                                config={"displayModeBar": False},
                            ),
                            style={"paddingTop": "0px"},
                        ),
                        sm=4,
                    )
                    for index in range(self.n_graphs)
                ],
                align="center",
                style={
                    "marginLeft": S.margin_side,
                    "marginRight": S.margin_side,
                    "marginTop": "50px",
                },
            )
        ]


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
        trims=((20, 40), 50, 70),
        graph_row=None,
        prefix_id="",
    ):
        if graph_row is None:
            graph_row = GraphRow()

        self.mri_data = mri_data
        self.hormones_df = hormones_df
        self.sliders = sliders
        self.graph_row = graph_row
        self.session_info = session_info

        super().__init__([sliders, graph_row, session_info], prefix_id)

        self.trims = [(trim, trim) if isinstance(trim, int) else trim for trim in trims]

    def _update_slider_lims(self):
        sliders = self.sliders

        sliders[0].var_def.default_value = sliders[0].var_def.default_value or 1
        sliders[0].var_def.max_value = min(
            sliders[0].var_def.max_value, len(self.mri_data)
        )
        for slider, trim in zip(sliders[1:], self.trims):
            var_def = slider.var_def
            min_value = var_def.min_value or trim[0]
            max_value = var_def.max_value or (self.mri_data[0].shape[0] - 1 - trim[1])
            var_def.default_value = (
                var_def.default_value or (max_value - min_value) // 2 + min_value
            )
            var_def.min_value = min_value
            var_def.max_value = max_value

    def to_dash(self):
        self._update_slider_lims()

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

        # TODO: move to other place?
        plot_ids = self.graph_row.ids()
        slider_ids = [slider.id for slider in self.sliders]
        create_update_nii_plot_basic(self.mri_data, plot_ids, slider_ids)

        session_ids = [elem.id for elem in self.session_info]
        create_update_session_info(self.hormones_df, slider_ids[0], self.session_info)

        return [instructions_text, plots, sliders_and_session]

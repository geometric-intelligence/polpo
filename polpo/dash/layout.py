import abc

import dash_bootstrap_components as dbc
from dash import html

from .style import STYLE as S


class Layout(abc.ABC):
    @abc.abstractmethod
    def to_dash(self, comps):
        pass


class TwoColumnLayout(Layout):
    def to_dash(self, comps):
        right, left = comps

        left_comp = left.to_dash()
        right_comp = dbc.Stack(
            right.to_dash(),
            gap=3,
        )

        return [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            left_comp,
                            style={"paddingTop": "0px"},
                        ),
                        sm=6,
                        width=900,
                    ),
                    dbc.Col(sm=3, width=100),
                    dbc.Col(right_comp, sm=3, width=500),
                ],
                align="center",
                style={
                    "marginLeft": S.margin_side,
                    "marginRight": S.margin_side,
                    "marginTop": "50px",
                },
            ),
        ]


class SwappedTwoColumnLayout(Layout):
    def to_dash(self, comps):
        left, right = comps

        right_comp = right.to_dash()
        left_comp = dbc.Stack(
            left.to_dash(),
            gap=3,
        )

        return [
            dbc.Row(
                [
                    dbc.Col(left_comp, sm=3, width=500),
                    dbc.Col(sm=3, width=100),
                    dbc.Col(
                        html.Div(
                            right_comp,
                            style={"paddingTop": "0px"},
                        ),
                        sm=6,
                        width=900,
                    ),
                ],
                align="center",
                style={
                    "marginLeft": S.margin_side,
                    "marginRight": S.margin_side,
                    "marginTop": "50px",
                },
            ),
        ]


class TwoRowLayout(Layout):
    def to_dash(self, comps):
        top, bottom = comps

        bottom_comp = bottom.to_dash()
        top_comp = dbc.Stack(
            top.to_dash(),
            gap=3,
        )

        row_style = {
            "marginLeft": S.margin_side,
            "marginRight": S.margin_side,
            "marginTop": "50px",
        }
        return [
            dbc.Row(
                [dbc.Col(top_comp, sm=3, width=500)],
                align="center",
                style=row_style,
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            bottom_comp,
                            style={"paddingTop": "0px"},
                        ),
                        sm=6,
                        width=900,
                    )
                ],
                align="center",
                style=row_style,
            ),
        ]


class SwappedTwoRowLayout(Layout):
    def to_dash(self, comps):
        bottom, top = comps

        top_comp = top.to_dash()
        bottom_comp = dbc.Stack(
            bottom.to_dash(),
            gap=3,
        )

        row_style = {
            "marginLeft": S.margin_side,
            "marginRight": S.margin_side,
            "marginTop": "50px",
        }
        return [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            top_comp,
                            style={"paddingTop": "0px"},
                        ),
                        sm=6,
                        width=900,
                    )
                ],
                align="center",
                style=row_style,
            ),
            dbc.Row(
                [dbc.Col(bottom_comp, sm=3, width=500)],
                align="center",
                style=row_style,
            ),
        ]

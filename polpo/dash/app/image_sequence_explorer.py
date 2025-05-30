import dash_bootstrap_components as dbc
from dash import Dash

import polpo.preprocessing.pd as ppd
from polpo.dash.components import (
    ComponentGroup,
    DepVar,
    ImageSeqExplorer,
    Slider,
)
from polpo.dash.style import update_style
from polpo.dash.variables import VarDef
from polpo.preprocessing import Map, Pipeline, Sorter, Truncater
from polpo.preprocessing.load.pregnancy import (
    DenseMaternalCsvDataLoader,
    PregnancyPilotMriLoader,
)
from polpo.preprocessing.mri import MriImageLoader


def _load_images():
    return Pipeline(
        steps=[
            PregnancyPilotMriLoader(as_dict=False),
            Sorter(),
            Truncater(value=2),  # For debugging
            Map(step=MriImageLoader(), n_jobs=1, verbose=1),
        ]
    )()


def _create_layout():

    images = _load_images()

    gest_week_ID = VarDef(
        id_="gestweekID", name="Gestation Week", min_value=0, max_value=41
    )

    image_seq_explorer = ImageSeqExplorer(
        
    )
    return dbc.Container(image_seq_explorer.to_dash())


def my_app():
    style = {
        "margin_side": "20px",
        "text_fontsize": "24px",
        "text_fontfamily": "Avenir",
        "title_fontsize": "40px",
        "space_between_sections": "70px",
        "space_between_title_and_content": "30px",
    }
    update_style(style)

    layout = _create_layout()

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )

    app.layout = layout

    app.run(
        debug=True,
        use_reloader=False,
        host="0.0.0.0",
        port="8050",
    )

import dash_bootstrap_components as dbc
from dash import Dash
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from polpo.dash.components import MeshExplorer, Slider
from polpo.dash.style import update_style
from polpo.dash.variables import VarDef
from polpo.models import Meshes2Comps, ObjectRegressor
from polpo.preprocessing import (
    IndexMap,
    Map,
    NestingSwapper,
    Pipeline,
    WrapInList,
)
from polpo.preprocessing import (
    dict as ppdict,
)
from polpo.preprocessing import pd as ppd
from polpo.preprocessing.load.pregnancy import (
    DenseMaternalCsvDataLoader,
    PregnancyPilotRegisteredMeshesLoader,
)
from polpo.preprocessing.mesh.io import TrimeshReader


def _load_session_week():
    hormones_df = Pipeline(
        steps=[
            DenseMaternalCsvDataLoader(pilot=True),
            ppd.Drop(labels=27),
        ]
    )()

    session_week_dict = Pipeline(
        steps=[
            ppd.ColumnsSelector(column_names="gestWeek"),
            ppd.SeriesToDict(),
        ],
    )(hormones_df)

    return session_week_dict


def _load_hipp_meshes():
    return Pipeline(
        steps=[
            PregnancyPilotRegisteredMeshesLoader(method="deformetrica", as_dict=True),
            ppdict.DictMap(step=TrimeshReader()),
        ]
    )()


def _load_data():
    return _load_session_week(), _load_hipp_meshes()


def _merge_session_week_meshes(session_week, registered_meshes):
    # data: hormones, registered_meshes
    pipeline = Pipeline(
        steps=[
            ppdict.DictMerger(),
            NestingSwapper(),
            IndexMap(index=0, step=Map(step=WrapInList())),
        ]
    )

    X, y = pipeline((session_week, registered_meshes))

    return X, y


def _instantiate_week_mesh_model(session_week, registered_meshes):
    X, y = _merge_session_week_meshes(session_week, registered_meshes)

    model = ObjectRegressor(
        model=LinearRegression(),
        objs2y=Meshes2Comps(
            dim_reduction=PCA(n_components=4),
            smoother=False,
        ),
    )

    model.fit(X, y)

    return model


def _create_week_inputs():
    gest_week = VarDef(
        id_="GestWeek",
        name="Gestational Week",
        min_value=0,
        max_value=36,
        default_value=15,
    )

    return Slider(gest_week)


def _hipp_week_single():
    session_week, registered_meshes = _load_data()
    model = _instantiate_week_mesh_model(session_week, registered_meshes)

    inputs = _create_week_inputs()

    mesh_explorer = MeshExplorer(model=model, inputs=inputs)

    return dbc.Container(mesh_explorer.to_dash())


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

    layout = _hipp_week_single()

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

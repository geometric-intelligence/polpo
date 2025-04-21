import dash_bootstrap_components as dbc
import numpy as np
from dash import Dash
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from polpo.dash.components import Graph, MeshExplorer, Slider
from polpo.dash.style import update_style
from polpo.dash.variables import VarDef
from polpo.models import DictMeshes2Comps, Meshes2Comps, ObjectRegressor
from polpo.plot.mesh import MeshesPlotter, MeshPlotter
from polpo.preprocessing import (
    IndexMap,
    ListSqueeze,
    Map,
    NestingSwapper,
    PartiallyInitializedStep,
    Pipeline,
    WrapInList,
)
from polpo.preprocessing import (
    dict as ppdict,
)
from polpo.preprocessing import pd as ppd
from polpo.preprocessing.load.pregnancy import (
    DenseMaternalCsvDataLoader,
    DenseMaternalMeshLoader,
    PregnancyPilotRegisteredMeshesLoader,
)
from polpo.preprocessing.mesh.conversion import TrimeshFromPv
from polpo.preprocessing.mesh.io import PvReader, TrimeshReader
from polpo.preprocessing.mesh.registration import PvAlign
from polpo.preprocessing.mri import segmtool2encoding


def _load_homornes_df():
    return Pipeline(
        steps=[
            DenseMaternalCsvDataLoader(pilot=True),
            ppd.Drop(labels=27),
        ]
    )()


def _load_session_week(hormones_df):
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


def _load_maternal_pilot():
    return Pipeline(
        steps=[
            DenseMaternalMeshLoader(subject_id=None, as_dict=True),
            ppdict.DictMap(step=PvReader()),
            PartiallyInitializedStep(
                Step=lambda target, max_iterations: ppdict.DictMap(
                    step=PvAlign(target=target, max_iterations=max_iterations)
                ),
                _target=lambda meshes: meshes[1],  # template mesh
                max_iterations=500,
            ),
            ppdict.DictMap(step=TrimeshFromPv()),
        ]
    )()


def _load_maternal_multiple(subject_id="01", tool="fsl"):
    encoding = segmtool2encoding(tool)

    struct_keys = encoding.structs

    mesh_loader = ppdict.HashWithIncoming(
        Map(
            PartiallyInitializedStep(
                Step=DenseMaternalMeshLoader,
                pass_data=False,
                subject_id=subject_id,
                _struct=lambda name: name.split("_")[-1],
                _left=lambda name: name.split("_")[0] == "L",
                as_dict=True,
            )
            + ppdict.DictMap(PvReader())
        )
    )

    prep_pipe = PartiallyInitializedStep(
        Step=lambda **kwargs: ppdict.DictMap(PvAlign(**kwargs) + TrimeshFromPv()),
        _target=lambda meshes: meshes[list(meshes.keys())[0]],
        max_iterations=500,
    )

    pipe = mesh_loader + ppdict.DictMap(prep_pipe)

    meshes = pipe(struct_keys)

    return meshes


def _merge_session_week_meshes(session_week, registered_meshes):
    pipeline = Pipeline(
        steps=[
            ppdict.DictMerger(),
            NestingSwapper(),
            IndexMap(index=0, step=Map(step=WrapInList())),
        ]
    )

    X, meshes = pipeline((session_week, registered_meshes))

    return X, meshes


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


def _merge_session_week_multi_meshes(session_week, registered_meshes):
    dict_pipe = (
        IndexMap(ppdict.NestedDictSwapper(), index=1)
        + ppdict.DictMerger()
        + NestingSwapper()
        + IndexMap(lambda x: np.array(x)[:, None], index=0)
        + IndexMap(ppdict.ListDictSwapper(), index=1)
    )

    # meshes_ : dict[list]
    X, meshes_ = dict_pipe([session_week, registered_meshes])

    return X, meshes_


def _instantiate_week_multi_mesh_model(session_week, registered_meshes):
    X, y = _merge_session_week_multi_meshes(session_week, registered_meshes)
    n_structs = len(y)

    pca = PCA(n_components=4)
    objs2y = DictMeshes2Comps(n_pipes=n_structs, dim_reduction=pca)

    model = ObjectRegressor(LinearRegression(fit_intercept=True), objs2y=objs2y)
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


Key2MeshLoader = {
    "hipp": _load_hipp_meshes,
    "maternal": _load_maternal_pilot,
    "multiple": _load_maternal_multiple,
}

Key2ModelInstantiator = {
    "hipp": _instantiate_week_mesh_model,
    "maternal": _instantiate_week_mesh_model,
    "multiple": _instantiate_week_multi_mesh_model,
}


def _week_mesh_layout(data="hipp"):
    hormones_df = _load_homornes_df()
    session_week = _load_session_week(hormones_df)

    inputs = _create_week_inputs()

    registered_meshes = Key2MeshLoader[data]()
    model = Key2ModelInstantiator[data](session_week, registered_meshes)

    postproc_pred = None
    graph = None
    if data == "multiple":
        plotter = MeshesPlotter([MeshPlotter() for _ in range(len(registered_meshes))])
        graph = Graph(id_="mesh-plot", plotter=plotter)

        postproc_pred = ppdict.DictMap(ListSqueeze()) + ppdict.DictToValuesList()

    mesh_explorer = MeshExplorer(
        model=model,
        inputs=inputs,
        graph=graph,
        postproc_pred=postproc_pred,
    )

    return dbc.Container(mesh_explorer.to_dash())


def my_app(data="hipp"):
    style = {
        "margin_side": "20px",
        "text_fontsize": "24px",
        "text_fontfamily": "Avenir",
        "title_fontsize": "40px",
        "space_between_sections": "70px",
        "space_between_title_and_content": "30px",
    }
    update_style(style)

    layout = _week_mesh_layout(data)

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

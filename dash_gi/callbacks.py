import plotly.graph_objects as go
from dash import Input, Output, callback


def create_view_model_update(input_view, output_view, model):
    empty_output = output_view.as_empty_output()

    @callback(
        output_view.as_output(),
        *input_view.as_input(),
    )
    def view_model_update(*args):
        if args[0] is None:
            return empty_output

        out = model.predict(args)
        return output_view.to_dash(out)


def create_show_mesh(mesh_data, graph, sliders):
    @callback(
        [Output(graph.id, "figure")],
        *[Input(slider.id, "drag_value") for slider in sliders],
    )
    def show_mesh(*args):
        if len(args) == 0:
            return [go.Figure()]

        fig = graph.to_dash(mesh=mesh_data)

        return (fig,)

from dash import Input, callback

from polpo.utils import unnest_list


def create_view_model_update(
    input_view, output_view, model, allow_duplicate=False, prevent_initial_call=False
):
    empty_output = output_view.as_empty_output()

    @callback(
        output_view.as_output(allow_duplicate=allow_duplicate),
        *input_view.as_input(),
        prevent_initial_call=prevent_initial_call,
    )
    def view_model_update(*args):
        if args[0] is None:
            return empty_output

        pred = model.predict(args)
        return output_view.to_dash(pred)


def create_button_toggler(toggle_id, hideable_components):
    n_components = len(hideable_components)

    @callback(
        unnest_list(
            [
                component.as_output(component_property="style")
                for component in hideable_components
            ]
        ),
        Input(toggle_id, "n_clicks"),
    )
    def button_toggler(n_clicks):
        show_index = n_clicks % n_components

        out = [{"display": "none"}] * n_components
        out[show_index] = {"display": "block"}

        return out


def create_button_toggler_for_view_model_update(
    input_views, output_view, models, toggle_id, hideable_components
):
    # NB: a simple merge of the two above to avoid chained callbacks
    empty_output = output_view.as_empty_output()

    n_components = len(hideable_components)

    inputs = []
    indices = [0]
    for input_view in input_views:
        input_ = input_view.as_input()
        indices.append(indices[-1] + len(input_))
        inputs.extend(input_)

    @callback(
        unnest_list(
            [
                component.as_output(component_property="style")
                for component in hideable_components
            ]
        )
        + output_view.as_output(),
        *([Input(toggle_id, "n_clicks")] + inputs),
    )
    def button_toggler(n_clicks, *args):
        out = [{"display": "none"}] * n_components

        if args[0] is None:
            return out + empty_output

        show_index = n_clicks % n_components
        model = models[show_index]
        model_args = args[indices[show_index] : indices[show_index + 1]]

        out[show_index] = {"display": "block"}
        pred = model.predict(model_args)

        return out + output_view.to_dash(pred)

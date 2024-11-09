import plotly.graph_objects as go
from dash import Dash, Input, Output, callback

from .plot import return_nii_plot


def create_update_nii_plot_basic(mri_data, plot_ids, slider_ids):
    # TODO: update to work with different kind of slices
    @callback(
        [Output(plot_id, "figure") for plot_id in plot_ids],
        *[Input(slider_id, "drag_value") for slider_id in slider_ids],
    )
    def update_nii_plot_basic(scan_number, x, y, z):
        """Update the nii plot based on the week and the x, y, z coordinates."""
        if scan_number is None:
            return [go.Figure()] * len(plot_ids)
        return return_nii_plot(mri_data[scan_number - 1], x, y, z)


def create_update_session_info(hormones_df, session_slider_id, session_info):
    @callback(
        [Output(elem.id, "children") for elem in session_info],
        Input(session_slider_id, "drag_value"),
    )
    def update_session_info(scan_number):
        if scan_number is None:
            return [""] * len(session_info)

        out = [session_info[0].to_dash(value=scan_number)]
        df_session = hormones_df.iloc[scan_number - 1]
        df_session.columns = hormones_df.columns

        for elem in session_info[1:]:
            value = df_session.get(elem.var_def.id)
            out.append(elem.to_dash(value))

        return out

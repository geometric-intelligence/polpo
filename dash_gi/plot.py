import numpy as np
import plotly.graph_objects as go

# TODO: move to other library?


def return_nii_plot(
    raw_mri_datum,
    x,
    y,
    z,
):  # week,
    """Return the nii plot based on the week and the x, y, z coordinates."""
    slice_0 = raw_mri_datum[x, :, :]
    slice_1 = raw_mri_datum[:, y, :]
    slice_2 = raw_mri_datum[:, :, z]

    # TODO: can be much simplified

    common_width = max(len(slice_0[:, 0]), len(slice_1[:, 0]), len(slice_2[:, 0]))
    common_height = max(len(slice_0[0]), len(slice_1[0]), len(slice_2[0]))

    slices = [slice_0, slice_1, slice_2]
    for i_slice, slice in enumerate([slice_0, slice_1, slice_2]):
        if len(slice[:, 0]) < common_width:
            diff = common_width - len(slice[:, 0])
            # slice = np.pad(slice, ((0, diff), (0, 0)), mode="constant")
            slice = np.pad(slice, ((diff // 2, diff // 2), (0, 0)), mode="constant")
            slices[i_slice] = slice
        if len(slice[0]) < common_height:
            diff = common_height - len(slice[0])
            # slice = np.pad(slice, ((0, 0), (0, diff)), mode="constant")
            slice = np.pad(slice, ((0, 0), (diff // 2, diff // 2)), mode="constant")
            slices[i_slice] = slice

    side_fig = plot_slice_as_plotly(
        slices[0], cmap="gray", title="Side View", x_label="Y", y_label="Z"
    )
    front_fig = plot_slice_as_plotly(
        slices[1], cmap="gray", title="Front View", x_label="X", y_label="Z"
    )
    top_fig = plot_slice_as_plotly(
        slices[2], cmap="gray", title="Top View", x_label="X", y_label="Y"
    )

    return side_fig, front_fig, top_fig


def plot_slice_as_plotly(
    one_slice, cmap="gray", title="Slice Visualization", x_label="X", y_label="Y"
):
    """Display an image slice as a Plotly figure."""
    # Create heatmap trace for the current slice
    heatmap_trace = go.Heatmap(z=one_slice.T, colorscale=cmap, showscale=False)

    width = int(len(one_slice[:, 0]) * 1.5)
    height = int(len(one_slice[0]) * 1.5)

    layout = go.Layout(
        title=title,
        title_x=0.5,
        xaxis=dict(title=x_label),
        yaxis=dict(title=y_label),
        width=width,
        height=height,
    )

    # Create a Plotly figure with the heatmap trace
    fig = go.Figure(data=heatmap_trace, layout=layout)

    # Update layout to adjust appearance
    # fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)

    return fig

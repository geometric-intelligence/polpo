import typer

app = typer.Typer()


from enum import Enum


class DataOptions(str, Enum):
    hipp = "hipp"
    maternal = "maternal"
    multiple = "multiple"


@app.command()
def mesh_explorer(
    data: DataOptions = DataOptions.hipp,
    hideable: bool = False,
    overlay: bool = False,
    week: bool = True,
    hormones: bool = True,
):
    """Launch mesh_explorer app."""
    from polpo.dash.app.mesh_explorer import my_app

    data = data.value

    if hideable and data != "multiple":
        # TODO: extend to single model?
        raise ValueError(
            "Cannot handle hiddeable for single structure and multiple models."
        )

    my_app(
        data=data,
        hideable=hideable,
        overlay=overlay,
        week=week,
        hormones=hormones,
    )


if __name__ == "__main__":
    app()

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
    switchable: bool = False,
    hideable: bool = False,
):
    """Launch mesh_explorer app."""
    from polpo.dash.app.mesh_explorer import my_app

    data = data.value

    if hideable and (data != "multiple" or not switchable):
        # TODO: extend to single model?
        raise ValueError(
            "Cannot handle hiddeable for single structure and multiple models."
        )

    my_app(data=data, switchable=switchable, hideable=hideable)


if __name__ == "__main__":
    app()

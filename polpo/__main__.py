import typer

app = typer.Typer()


from enum import Enum


class DataOptions(str, Enum):
    hipp = "hipp"
    maternal = "maternal"
    multiple = "multiple"


@app.command()
def mesh_explorer(data: DataOptions = DataOptions.hipp, switchable: bool = False):
    """Launch mesh_explorer app."""
    from polpo.dash.app.mesh_explorer import my_app

    my_app(data=data.value, switchable=switchable)


if __name__ == "__main__":
    app()

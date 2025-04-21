import typer

app = typer.Typer()


@app.command()
def mesh_explorer():
    """Launch mesh_explorer app."""
    from polpo.dash.app.mesh_explorer import my_app

    my_app()


if __name__ == "__main__":
    app()

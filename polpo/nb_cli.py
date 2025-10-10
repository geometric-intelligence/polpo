from typing import List

import nbformat
import typer

import polpo.nbformat.utils as pnbformatu
import polpo.utils as putils

app = typer.Typer()


# TODO: add logging?


class NbItem:
    def __init__(self, nb, path, modified=False):
        self.nb = nb
        self.path = path
        self.modified = modified

    @classmethod
    def from_path(cls, path):
        nb = nbformat.read(path, as_version=4)
        return cls(nb, path)

    def write(self, path=None):
        modified = True
        if path is None:
            path = self.path
            modified = self.modified

        if modified:
            nbformat.write(self.nb, path)

        return path


def _expand_notebook_names(names):
    """Expand a list of filename/glob names into a de-duplicated list of Paths."""

    paths = putils.expand_path_names(names)
    return [path for path in paths if path.suffix == ".ipynb"]


def _load_notebooks(names):
    paths = _expand_notebook_names(names)

    return [NbItem.from_path(path) for path in paths]


def _write_notebooks(notebooks):
    for notebook in notebooks:
        notebook.write()


@app.command()
def rm_md_key(
    notebooks: List[str],
    key: str,
):
    notebooks = _load_notebooks(notebooks)

    for notebook in notebooks:
        notebook.modified = pnbformatu.remove_metadata_key(notebook.nb, key)

    _write_notebooks(notebooks)


@app.command()
def mv_md_key(
    notebooks: List[str],
    old_key: str,
    new_key: str,
):
    notebooks = _load_notebooks(notebooks)

    for notebook in notebooks:
        notebook.modified = pnbformatu.rename_metadata_key(
            notebook.nb, old_key, new_key
        )

    _write_notebooks(notebooks)


@app.command()
def add_md_vals(
    notebooks: List[str],
    vals: str,
    key: str = "tags",
):
    notebooks = _load_notebooks(notebooks)

    if "," in vals:
        vals = vals.replace(" ", "").split(",")
    else:
        vals = [vals]

    for notebook in notebooks:
        notebook.modified = pnbformatu.add_metadata_values(notebook.nb, vals, key=key)

    _write_notebooks(notebooks)

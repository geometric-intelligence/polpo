import logging
import sys
from typing import List

import nbformat
import typer

import polpo.nb.utils as pnbutils
import polpo.utils as putils

app = typer.Typer()


def _set_logger(logging_level):
    logging.basicConfig(
        level=logging_level, format="%(levelname)s:polpo-nb:%(message)s"
    )


class NbItem:
    def __init__(self, nb, path):
        self.nb = nb
        self.path = path
        self._modified = False

    @property
    def modified(self):
        return self._modified

    @modified.setter
    def modified(self, val):
        if val:
            logging.info(f"Modified {self.path.as_posix()}.")

        self._modified = val

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
    """Expand a list of filename/glob names into a de-duplicated list of Paths.

    Notes
    -----
    * shell resolves globs before they reach here
    """
    paths = putils.expand_path_names(names)
    return [path for path in paths if path.suffix == ".ipynb"]


def _load_notebooks(names):
    paths = _expand_notebook_names(names)

    return [NbItem.from_path(path) for path in paths]


def _write_notebooks(notebooks):
    for notebook in notebooks:
        notebook.write()


def _process_cs_string(string):
    if "," in string:
        string = string.replace(" ", "").split(",")
    else:
        string = [string]

    return string


def _collection_as_str(data, sep=","):
    return f"{sep}".join(data)


def _modify_with_vals(notebooks, vals, key, func):
    notebooks = _load_notebooks(notebooks)

    vals = _process_cs_string(vals)

    for notebook in notebooks:
        notebook.modified = func(notebook.nb, vals, key=key)

    _write_notebooks(notebooks)


@app.command()
def rm_md_key(
    notebooks: List[str],
    key: str,
    logging_level: int = 20,
):
    _set_logger(logging_level)

    notebooks = _load_notebooks(notebooks)

    for notebook in notebooks:
        notebook.modified = pnbutils.remove_metadata_key(notebook.nb, key)

    _write_notebooks(notebooks)


@app.command()
def mv_md_key(
    notebooks: List[str],
    old_key: str,
    new_key: str,
    logging_level: int = 20,
):
    _set_logger(logging_level)

    notebooks = _load_notebooks(notebooks)

    for notebook in notebooks:
        notebook.modified = pnbutils.rename_metadata_key(notebook.nb, old_key, new_key)

    _write_notebooks(notebooks)


@app.command()
def add_md_vals(
    notebooks: List[str],
    vals: str,
    key: str = "tags",
    logging_level: int = 20,
):
    _set_logger(logging_level)

    _modify_with_vals(notebooks, vals, key, pnbutils.add_metadata_values)


@app.command()
def rm_md_vals(
    notebooks: List[str],
    vals: str,
    key: str = "tags",
    logging_level: int = 20,
):
    _set_logger(logging_level)

    _modify_with_vals(notebooks, vals, key, pnbutils.remove_metadata_values)


@app.command()
def get_md_vals(
    notebooks: List[str],
    key: str = "tags",
    tab_size: int = 3,
):
    notebooks = _load_notebooks(notebooks)

    data = {}
    for notebook in notebooks:
        key_vals = pnbutils.get_metadata_values(notebook.nb, key=key)
        if key_vals is not None:
            data[notebook.path] = key_vals

    text = ""
    tab = " " * tab_size
    for path, vals in data.items():
        text += f"{path.as_posix()}\n"
        text += f"{tab}{_collection_as_str(vals)}\n\n"

    sys.stdout.write(text)


@app.command()
def get_md_keys(
    notebooks: List[str],
    tab_size: int = 3,
    exclude: str = "kernelspec,language_info",
):
    notebooks = _load_notebooks(notebooks)
    exclude = set(_process_cs_string(exclude))

    data = {}
    for notebook in notebooks:
        keys = pnbutils.get_metadata_keys(notebook.nb)
        filtered_keys = [key for key in keys if key not in exclude]
        if filtered_keys:
            data[notebook.path] = filtered_keys

    text = ""
    tab = " " * tab_size
    for path, vals in data.items():
        text += f"{path.as_posix()}\n"
        text += f"{tab}{_collection_as_str(vals)}\n\n"

    sys.stdout.write(text)


@app.command()
def mdkey_contains_val(
    notebooks: List[str],
    val: str,
    key: str = "tags",
    sep: str = "\n",
):
    notebooks = _load_notebooks(notebooks)

    filtered = filter(
        lambda notebook: pnbutils.key_contains_val(notebook.nb, val, key=key),
        notebooks,
    )

    text = f"{sep}".join([nb.path.as_posix() for nb in filtered])

    if text == "":
        return

    text += "\n"

    sys.stdout.write(text)


@app.command()
def has_mdkey(
    notebooks: List[str],
    key: str = "tags",
    sep: str = "\n",
):
    notebooks = _load_notebooks(notebooks)

    filtered = filter(
        lambda notebook: key in pnbutils.get_metadata_keys(notebook.nb),
        notebooks,
    )

    text = f"{sep}".join([nb.path.as_posix() for nb in filtered])

    if text == "":
        return

    text += "\n"

    sys.stdout.write(text)

import os
import warnings
from pathlib import Path

import nbformat


def _handle_tutorials(config, doc_notebooks_path):
    tutorials_path = Path.cwd() / config.nbsymlink_notebooks_dir / "examples"

    os.symlink(
        tutorials_path,
        doc_notebooks_path / "tutorials",
        target_is_directory=True,
    )


def _handle_how_tos(config, tags, doc_notebooks_path, all_notebooks=True):
    # TODO: config out?
    how_tos_path = Path.cwd() / config.nbsymlink_notebooks_dir / "how_to"
    doc_how_tos_path = doc_notebooks_path / "how_to"

    os.makedirs(doc_how_tos_path, exist_ok=True)

    # TODO: from config
    if all_notebooks:
        tags += ["all"]
    notebooks_by_topic = {tag: [] for tag in tags}

    # TODO: collect if None
    # TODO: need to make consistent with rst
    for file in how_tos_path.iterdir():
        if file.suffix != ".ipynb":
            continue

        # TODO: make it optional
        if all_notebooks:
            notebooks_by_topic["all"].append(file)

        metadata = nbformat.read(file, as_version=4).metadata

        # TODO: tag by default? make it configurable?
        for topic in metadata.get("docs_topics", []):
            if topic not in notebooks_by_topic:
                warnings.warn(f"{topic} does not exist.")
                continue

            notebooks_by_topic[topic].append(file)

    for topic, notebooks in notebooks_by_topic.items():
        examples_topic_path = doc_how_tos_path / topic

        os.makedirs(examples_topic_path, exist_ok=True)
        for notebook in notebooks:
            os.symlink(
                notebook,
                examples_topic_path / notebook.name,
                target_is_directory=False,
            )


def run(app, config):
    doc_notebooks_path = Path.cwd() / "docs/_notebooks"
    tags = config.nbsymlink_tags
    all_notebooks = config.nbsymlink_all
    os.makedirs(doc_notebooks_path, exist_ok=True)

    # TODO: handle them similarly?
    _handle_tutorials(config, doc_notebooks_path)
    _handle_how_tos(config, tags, doc_notebooks_path, all_notebooks=all_notebooks)


def setup(app):
    """Entry point for the Sphinx extension."""
    app.add_config_value("nbsymlink_notebooks_dir", "default_value", "notebooks")
    app.add_config_value("nbsymlink_tags", "default_value", None, types=list)
    app.add_config_value("nbsymlink_all", "default_value", True, types=bool)

    app.connect("config-inited", run)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

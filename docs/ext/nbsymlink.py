import json
import os
import warnings
from pathlib import Path


def _handle_tutorials(config, doc_notebooks_path):
    tutorials_path = Path.cwd() / config.nbsymlink_notebooks_dir / "examples"

    os.symlink(
        tutorials_path,
        doc_notebooks_path / "tutorials",
        target_is_directory=True,
    )


def _handle_how_tos(config, doc_notebooks_path):
    how_tos_path = Path.cwd() / config.nbsymlink_notebooks_dir / "how_to"
    doc_how_tos_path = doc_notebooks_path / "how_to"

    os.makedirs(doc_how_tos_path, exist_ok=True)

    topics = ["data_loading", "mesh", "mri"] + ["all"]
    notebooks_by_topic = {topic: [] for topic in topics}
    for file in how_tos_path.iterdir():
        if file.suffix != ".ipynb":
            continue

        with open(file, "r", encoding="utf8") as file_:
            metadata = json.load(file_).get("metadata")

        notebooks_by_topic["all"].append(file)

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


def create_notebooks_folder(app, config):
    doc_notebooks_path = Path.cwd() / "docs/_notebooks"
    os.makedirs(doc_notebooks_path, exist_ok=True)

    _handle_tutorials(config, doc_notebooks_path)
    _handle_how_tos(config, doc_notebooks_path)


def setup(app):
    """Entry point for the Sphinx extension."""
    app.add_config_value("nbsymlink_notebooks_dir", "default_value", "notebooks")
    app.connect("config-inited", create_notebooks_folder)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

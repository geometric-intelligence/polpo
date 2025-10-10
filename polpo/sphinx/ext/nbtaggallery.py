import nbformat

from polpo.sphinx.utils import create_directive_str


def _group_by_tags(notebooks_path, include_path, tags=(), auto=False):
    # include_path for relative paths
    grouped_notebooks = {tag: [] for tag in tags}

    for file in notebooks_path.rglob("*.ipynb"):
        metadata = nbformat.read(file, as_version=4).metadata

        # TODO: use tags instead
        for tag in metadata.get("docs_topics", []):
            if tag not in grouped_notebooks:
                if not auto:
                    continue

                grouped_notebooks[tag] = {}

            grouped_notebooks[tag].append(
                file.relative_to(include_path, walk_up=True).as_posix()
            )

    return grouped_notebooks


def _create_galleries(
    notebooks_path, srcdir, include_path, tags=(), auto=False, tag_captions=None
):
    if tag_captions is None:
        tag_captions = {}

    grouped_notebooks = _group_by_tags(
        notebooks_path,
        include_path,
        tags=tags,
        auto=auto,
    )

    galleries_str_ls = []
    for tag, filenames in grouped_notebooks.items():
        if len(filenames) == 0:
            continue

        caption = tag_captions.get(tag)
        if caption is None:
            caption = tag.replace("_", " ").capitalize()

        text = caption + f"\n{'='*len(caption)}\n\n"
        text += create_directive_str("nblinkgallery", filenames)

        text += "\n\n"
        text += create_directive_str(
            "toctree",
            filenames,
            caption=caption,
            hidden="",
            maxdepth=1,
        )
        galleries_str_ls.append(text)

    galleries_str = "\n\n\n".join(galleries_str_ls)

    # TODO: do it with a directive instead, e.g. like autosummary?
    galleries_path = (
        srcdir / "_generated" / "nbgalleries" / f"{notebooks_path.name}.rst"
    )
    galleries_path.parent.mkdir(parents=True, exist_ok=True)
    with open(galleries_path, "w") as file:
        file.write(galleries_str)


def run(app, config):
    _create_galleries(
        app.srcdir / "_generated/notebooks" / "how_to",
        include_path=app.srcdir / config.nbtaggallery_include_path,
        srcdir=app.srcdir,
        tags=config.nbtaggallery_tags,
        auto=config.nbtaggallery_auto_tag,
        tag_captions=config.nbtaggallery_tag_captions,
    )


def setup(app):
    """Entry point for the Sphinx extension."""
    app.add_config_value("nbtaggallery_tags", "default_value", (), types=list)
    app.add_config_value("nbtaggallery_auto_tag", "default_value", False, types=bool)
    app.add_config_value("nbtaggallery_tag_captions", "default_value", None, types=dict)
    app.add_config_value("nbtaggallery_include_path", "default_value", "how_to")

    app.connect("config-inited", run)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

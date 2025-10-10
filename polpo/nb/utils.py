def remove_metadata_key(notebook, key):
    if key in notebook.metadata:
        notebook.metadata.pop(key)
        return True

    return False


def rename_metadata_key(notebook, old_key, new_key):
    metadata = notebook.metadata
    if old_key in metadata:
        metadata[new_key] = notebook.metadata.pop(old_key)
        return True

    return False


def add_metadata_values(notebook, values, key="tags"):
    metadata = notebook.metadata

    existing_values = metadata.get(key, [])

    modified = False
    for value in values:
        if value not in existing_values:
            modified = True
            existing_values.append(value)

    metadata[key] = existing_values

    return modified


def remove_metadata_values(notebook, values, key="tags"):
    metadata = notebook.metadata

    if key not in metadata:
        return False

    existing_values = metadata.get(key)

    modified = False

    for value in values:
        if value in existing_values:
            modified = True
            existing_values.remove(value)

    metadata[key] = existing_values

    return modified


def get_metadata_values(notebook, key="tags"):
    metadata = notebook.metadata
    return metadata.get(key, None)


def get_metadata_keys(notebook):
    metadata = notebook.metadata
    return list(metadata.keys())


def key_contains_val(notebook, val, key="tags"):
    metadata = notebook.metadata

    if key not in metadata:
        return False

    return val in metadata.get(key)

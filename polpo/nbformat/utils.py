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

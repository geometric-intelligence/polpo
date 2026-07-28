import numpy as np

_KEY_ENCODERS = {
    str: str,
    int: str,
    float: repr,
    bool: lambda value: "true" if value else "false",
}

_KEY_DECODERS = {
    "str": str,
    "int": int,
    "float": float,
    "bool": lambda value: value == "true",
}

_KEY_TYPE_NAMES = {key_type: key_type.__name__ for key_type in _KEY_ENCODERS}


def _encode_keys(keys):
    keys = list(keys)

    if not keys:
        raise ValueError("Cannot save empty keys.")

    keys_are_tuples = isinstance(keys[0], tuple)
    normalized = [key if keys_are_tuples else (key,) for key in keys]

    arity = len(normalized[0])
    key_types = tuple(type(value) for value in normalized[0])

    for key in normalized:
        if len(key) != arity:
            raise ValueError("All tuple keys must have the same length.")

        if tuple(type(value) for value in key) != key_types:
            raise TypeError("All keys must have the same component types.")

    unsupported = [key_type for key_type in key_types if key_type not in _KEY_ENCODERS]
    if unsupported:
        names = ", ".join(key_type.__name__ for key_type in unsupported)
        raise TypeError(f"Unsupported key types: {names}.")

    encoded = [
        [_KEY_ENCODERS[key_type](value) for key_type, value in zip(key_types, key)]
        for key in normalized
    ]

    return {
        "keys": np.asarray(encoded, dtype=str),
        "key_types": np.asarray(
            [_KEY_TYPE_NAMES[key_type] for key_type in key_types],
            dtype=str,
        ),
        "keys_are_tuples": np.asarray(keys_are_tuples),
    }


def _decode_keys(archive):
    encoded_keys = archive["keys"].tolist()
    type_names = archive["key_types"].tolist()
    keys_are_tuples = bool(archive["keys_are_tuples"])

    decoders = [_KEY_DECODERS[type_name] for type_name in type_names]

    keys = [
        tuple(decoder(value) for decoder, value in zip(decoders, encoded_key))
        for encoded_key in encoded_keys
    ]

    if not keys_are_tuples:
        keys = [key[0] for key in keys]

    return keys


def save_indexed_array(path, keys, data):
    encoded_keys = _encode_keys(keys)

    np.savez_compressed(
        path,
        **encoded_keys,
        data=data,
        representation="condensed",
        format_version=1,
    )


def load_indexed_array(path):
    with np.load(path, allow_pickle=False) as data:
        return _decode_keys(data), data["data"]


def save_dict_as_array(path, data):
    """Save a key-indexed dictionary as an array and ordered keys.

    Parameters
    ----------
    path : path-like
        Output ``.npz`` path.
    data : mapping
        Mapping from keys to values. All values must have compatible shapes.
        Dictionary insertion order determines the array indexing.
    """
    return save_indexed_array(
        path,
        keys=list(data),
        data=np.stack(list(data.values())),
    )


def load_dict(path):
    keys, data = load_indexed_array(path)
    return dict(zip(keys, data))

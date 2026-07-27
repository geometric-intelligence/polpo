import numpy as np

from polpo.utils.np import triu_vec_to_sym

# TODO: move numpy-encoding somewhere numpy related and add writer to PairwiseDists?

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


def sort_dist_mat(mat):
    # checks for global isolation
    dist_norms = np.linalg.norm(mat, axis=-1)
    sorted_idx = np.argsort(-dist_norms, axis=-1)

    perm_dists = mat[sorted_idx][:, sorted_idx]

    return perm_dists, sorted_idx


def knn_scores(mat, k=5):
    # mat: (n, n) distance matrix
    idx = np.argsort(mat, axis=1)
    knn = np.take_along_axis(mat, idx[:, 1 : k + 1], axis=1)
    return knn.mean(axis=1)


class PairwiseDistances:
    def __init__(self, keys, data):
        self.keys = keys
        self.data = data

        self._validate()

        self._key_to_index = {key: index for index, key in enumerate(self.keys)}

    def _validate(self):
        expected = len(self.keys) * (len(self.keys) - 1) // 2

        if len(self.data) != expected:
            raise ValueError(
                f"Expected {expected} distances, " f"got {len(self.data)}."
            )

    @property
    def matrix(self):
        return triu_vec_to_sym(self.data)

    def index(self, key):
        return self._key_to_index[key]

    @staticmethod
    def _pair_index(i, j, n):
        if i == j:
            return None

        if i > j:
            i, j = j, i

        return n * i - i * (i + 1) // 2 + (j - i - 1)

    def get(self, key_a, key_b):
        i = self.index(key_a)
        j = self.index(key_b)

        if i == j:
            return 0.0

        index = self._pair_index(i, j, len(self.keys))
        return self.data[index]

    def save(self, path):
        encoded_keys = _encode_keys(self.keys)

        np.savez_compressed(
            path,
            **encoded_keys,
            data=self.data,
            representation="condensed",
            format_version=1,
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(
                keys=_decode_keys(data),
                data=data["data"],
            )


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

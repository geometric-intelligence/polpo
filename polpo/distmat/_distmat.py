import numpy as np

from polpo.numpy.io import load_indexed_array, save_indexed_array
from polpo.utils.np import triu_vec_to_sym


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
        return save_indexed_array(path, self.keys, self.data)

    @classmethod
    def load(cls, path):
        keys, data = load_indexed_array(path)
        return cls(keys=keys, data=data)

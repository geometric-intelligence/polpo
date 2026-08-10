import numpy as np

from polpo.dataset import Dataset
from polpo.numpy.io import load_indexed_array, save_indexed_array
from polpo.utils.np import triu_vec_to_sym

from .plot import plot_dist_mat


def sort_by_row_norm(mat, descending=True):
    # checks for global isolation
    row_norms = np.linalg.norm(mat, axis=-1)

    signal = -1.0 if descending else 1.0
    sorted_idx = np.argsort(signal * row_norms, axis=-1)

    perm_mat = mat[np.ix_(sorted_idx, sorted_idx)]

    return perm_mat, sorted_idx


def knn_scores(mat, k=5):
    # mat: (n, n) distance matrix
    idx = np.argsort(mat, axis=1)
    knn = np.take_along_axis(mat, idx[:, 1 : k + 1], axis=1)
    return knn.mean(axis=1)


class PairwiseDistances:
    def __init__(self, labels, data):
        self.labels = labels
        self.data = data

        self._validate()

        self._label_to_index = {label: index for index, label in enumerate(self.labels)}

    def _validate(self):
        expected = len(self.labels) * (len(self.labels) - 1) // 2

        if len(self.data) != expected:
            raise ValueError(f"Expected {expected} distances, got {len(self.data)}.")

    @property
    def matrix(self):
        return triu_vec_to_sym(self.data)

    @classmethod
    def from_matrix(cls, labels, matrix):
        indices = np.triu_indices(len(labels), k=1)
        return cls(labels, matrix[indices])

    @property
    def pairs(self):
        return [
            (self.labels[i], self.labels[j])
            for i in range(len(self.labels))
            for j in range(i + 1, len(self.labels))
        ]

    def items(self):
        return zip(self.pairs, self.data)

    def as_dataset(self):
        return Dataset(dict(self.items()))

    def _index(self, label):
        return self._label_to_index[label]

    @staticmethod
    def _pair_index(i, j, n):
        if i == j:
            return None

        if i > j:
            i, j = j, i

        return n * i - i * (i + 1) // 2 + (j - i - 1)

    def get(self, label_a, label_b):
        i = self._index(label_a)
        j = self._index(label_b)

        if i == j:
            return 0.0

        index = self._pair_index(i, j, len(self.labels))
        return self.data[index]

    def save(self, path):
        return save_indexed_array(path, self.labels, self.data)

    @classmethod
    def load(cls, path):
        labels, data = load_indexed_array(path)
        return cls(labels=labels, data=data)

    def sort_by_row_norm(self, descending=True):
        perm_mat, sorted_idx = sort_by_row_norm(self.matrix)
        labels = [self.labels[index] for index in sorted_idx]
        return self.__class__.from_matrix(labels, perm_mat)

    def plot(self, **kwargs):
        return plot_dist_mat(self, **kwargs)

    def select(self, labels):
        data = [self.get(a, b) for i, a in enumerate(labels) for b in labels[i + 1 :]]

        return self.__class__(labels, np.array(data))

    def filter_labels(self, predicate):
        return self.select([label for label in self.labels if predicate(label)])

    def map_labels(self, func):
        return self.__class__(
            labels=[func(label) for label in self.labels],
            data=self.data,
        )

import abc
import copy

import numpy as np


class Model(abc.ABC):
    @abc.abstractmethod
    def predict(self, X):
        pass


class ModelFactory(abc.ABC):
    @abc.abstractmethod
    def create(self):
        # returns a `Model`
        pass


class ConstantOutput(Model):
    # for debugging
    def __init__(self, value):
        super().__init__()
        self.value = value

    def predict(self, X=None):
        return self.value


class ListLookup(Model):
    def __init__(self, data, tar=0):
        super().__init__()
        self.data = data
        self.tar = tar

    def predict(self, X):
        # NB: expects a (int,)
        return self.data[X[0] - self.tar]


class PdDfLookup(Model):
    def __init__(self, df, output_keys, tar=0):
        super().__init__()
        self.df = df
        self.output_keys = output_keys
        self.tar = tar

    def predict(self, X):
        # NB: expects a (int,)

        # TODO: make it input key based instead?
        df_session = self.df.iloc[X[0] - self.tar]
        df_session.columns = self.df.columns

        return [df_session.get(key) for key in self.output_keys]


class MriSlicesLookup(Model):
    def __init__(self, data, index_tar=1, index_ordering=(0, 1, 2)):
        self.data = data
        self.index_tar = index_tar
        self.index_ordering = index_ordering

    def predict(self, X):
        index, *slice_indices = X

        datum = self.data[index - self.index_tar]

        slices = []
        for index, slice_index in zip(self.index_ordering, slice_indices):
            slicing_indices = [slice(None)] * 3
            slicing_indices[index] = slice_index
            slices.append(datum[tuple(slicing_indices)])

        common_width = max([len(slice_[:, 0]) for slice_ in slices])
        common_height = max([len(slice_[0]) for slice_ in slices])

        for i_slice, slice_ in enumerate(slices):
            if len(slice_[:, 0]) < common_width:
                diff = common_width - len(slice_[:, 0])
                slice_ = np.pad(
                    slice_, ((diff // 2, diff // 2), (0, 0)), mode="constant"
                )
                slices[i_slice] = slice_
            if len(slice_[0]) < common_height:
                diff = common_height - len(slice_[0])
                slice_ = np.pad(
                    slice_, ((0, 0), (diff // 2, diff // 2)), mode="constant"
                )
                slices[i_slice] = slice_

        return slices


class LinearMeshVertexScaling(Model):
    def __init__(self, mesh):
        self.mesh = mesh
        self._updatable_mesh = copy.deepcopy(mesh)

    def predict(self, X):
        # NB: expects a (int,)
        self._updatable_mesh.vertices = X[0] * self.mesh.vertices
        return self._updatable_mesh

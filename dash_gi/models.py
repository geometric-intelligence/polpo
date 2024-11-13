import abc

import numpy as np


class Model(abc.ABC):
    # TODO: a model may be a `Representation` (following dash-vtk)

    @abc.abstractmethod
    def predict(self, X):
        pass


class PdDfLookupModel(Model):
    def __init__(self, df, output_keys, tar=0):
        self.df = df
        self.output_keys = output_keys
        self.tar = tar

    def predict(self, X):
        # NB: expects a (int,)

        # TODO: make it input key based instead?
        df_session = self.df.iloc[X[0] - self.tar]
        df_session.columns = self.df.columns

        return [df_session.get(key) for key in self.output_keys]


class MriSlices(Model):
    def __init__(self, data, index_tar=1):
        self.data = data
        self.index_tar = index_tar

    def predict(self, X):
        index, x, y, z = X

        datum = self.data[index - self.index_tar]
        slice_0 = datum[x, :, :]
        slice_1 = datum[:, y, :]
        slice_2 = datum[:, :, z]

        common_width = max(len(slice_0[:, 0]), len(slice_1[:, 0]), len(slice_2[:, 0]))
        common_height = max(len(slice_0[0]), len(slice_1[0]), len(slice_2[0]))

        slices = [slice_0, slice_1, slice_2]
        for i_slice, slice_ in enumerate([slice_0, slice_1, slice_2]):
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

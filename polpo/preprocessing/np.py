import numpy as np

from .base import PreprocessingStep


class ToArray(PreprocessingStep):
    def apply(self, data):
        return np.array(data)


class AtLeast2d(PreprocessingStep):
    def apply(self, data):
        return np.atleast_2d(data)


class Squeeze(PreprocessingStep):
    def apply(self, data):
        return np.squeeze(data)


class Stack(PreprocessingStep):
    def __init__(self, axis=0):
        super().__init__()
        self.axis = axis

    def apply(self, data):
        return np.stack(data, axis=self.axis)


class Reshape(PreprocessingStep):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def apply(self, data):
        return np.reshape(data, self.shape)


class FlattenButFirst(PreprocessingStep):
    def apply(self, data):
        return np.reshape(data, (data.shape[0], -1))


class Complex2RealsRepr(PreprocessingStep):
    def apply(self, cnumber):
        return np.hstack([cnumber.real, cnumber.imag])


class RealsRepr2Complex(PreprocessingStep):
    def apply(self, rnumber):
        half_index = rnumber.shape[-1] // 2
        return rnumber[..., :half_index] + 1j * rnumber[..., half_index:]

import numpy as np

from .base import PreprocessingStep


class ToArray(PreprocessingStep):
    def apply(self, data):
        return np.array(data)


class AtLeast2d(PreprocessingStep):
    def apply(self, data):
        return np.atleast_2d(data)


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

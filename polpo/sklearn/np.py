"""Transformer mixins handling numpy arrays."""

from sklearn.preprocessing import FunctionTransformer

from polpo.preprocessing.np import FlattenButFirst, Reshape


class InvertibleFlattenButFirst(FunctionTransformer):
    """Function transformer for for flattening array but first axis."""

    def __init__(self):
        super().__init__(
            func=FlattenButFirst(),
            inverse_func=Reshape(None),
            check_inverse=False,
        )

    def fit(self, X, y=None):
        # y is ignored
        self.inverse_func.shape = (-1,) + X.shape[1:]
        return self

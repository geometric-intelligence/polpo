from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer

from dash_gi.preprocessing import Pipeline


class TransformerAdapter(TransformerMixin):
    def __init__(self, step):
        self.step = step

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.step(X)


class InvertiblePipeline(FunctionTransformer):
    def __init__(self, steps):
        # steps: list[FunctionTransformer]

        self.steps = steps
        super().__init__(
            func=Pipeline(steps=[step.transform for step in self.steps]),
            inverse_func=Pipeline(
                steps=[step.inverse_transform for step in reversed(self.steps)]
            ),
            check_inverse=False,
        )

    def _fit(self, X, y=None):
        for step in self.steps:
            X = step.fit_transform(X, y)
        return X

    def fit(self, X, y=None):
        self._fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        return self._fit(X, y)

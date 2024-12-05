import abc

from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer

from polpo.preprocessing import Pipeline


class TransformerAdapter(TransformerMixin):
    def __init__(self, step):
        self.step = step

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.step(X)


class _PipelineMixin(abc.ABC):
    def __init__(self, steps, **kwargs):
        self.steps = steps
        super().__init__(**kwargs)

    def _fit(self, X, y=None):
        for step in self.steps:
            X = step.fit_transform(X, y)
        return X

    def fit(self, X, y=None):
        self._fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        return self._fit(X, y)


class AdapterPipeline(_PipelineMixin, TransformerMixin):
    def __init__(self, steps):
        for index, step in enumerate(steps):
            if not isinstance(step, TransformerMixin):
                steps[index] = TransformerAdapter(step)

        super().__init__(steps=steps)

    def transform(self, X):
        for step in self.steps:
            X = step.transform(X)
        return X


class InvertiblePipeline(_PipelineMixin, FunctionTransformer):
    def __init__(self, steps):
        # steps: list[FunctionTransformer]
        super().__init__(
            steps=steps,
            func=Pipeline(steps=[step.transform for step in steps]),
            inverse_func=Pipeline(
                steps=[step.inverse_transform for step in reversed(steps)]
            ),
            check_inverse=False,
        )

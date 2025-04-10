"""Adapters for sklearn."""

from collections.abc import Iterable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class TransformerAdapter(TransformerMixin, BaseEstimator):
    """Adapts a step with TransformerMixin behavior.

    Makes any callable compatible with `sklearn.TransformerMixin`.
    Assumes callable does not need to be fitted.

    Parameters
    ----------
    step : callable
        Step to be adapted.
    """

    def __init__(self, step):
        self.step = step
        super().__init__()

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        return self.step(X)


class AdapterPipeline(Pipeline):
    """sklearn compatible pipeline.

    Names and adapts steps if needed.
    Syntax sugar for `sklearn.Pipeline` without the
    need to name steps, and with the ability of having
    callables as steps.

    Parameters
    ----------
    steps : list
        Steps to be adapted.
    """

    def __init__(self, steps):
        self._unadapted_steps = steps

        adapted_steps = []
        for index, step in enumerate(steps):
            if not isinstance(step, Iterable):
                step_name = f"step_{index}"
                step = (step_name, step)

            if not isinstance(step[1], TransformerMixin):
                step = (step[0], TransformerAdapter(step[1]))

            adapted_steps.append(step)

        super().__init__(steps=adapted_steps)

    def __sklearn_clone__(self):
        return AdapterPipeline(steps=self._unadapted_steps)

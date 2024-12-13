from collections.abc import Iterable

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline


class TransformerAdapter(TransformerMixin):
    def __init__(self, step):
        self.step = step

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.step(X)


class AdapterPipeline(Pipeline):
    # names and adapts steps
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

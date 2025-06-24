"""Adapters for sklearn."""

from collections.abc import Iterable

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.utils.validation import check_is_fitted


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


class MapTransformer(TransformerMixin, BaseEstimator):
    # TODO: create one with base step?
    # TODO: allow parallel?

    def __init__(self, par_steps):
        self.par_steps = par_steps
        super().__init__()

    def fit(self, X, y=None):
        self.is_fitted_ = True
        for step, x in zip(self.par_steps, X):
            step.fit(x)

        return self

    def transform(self, X):
        return [step.transform(x) for step, x in zip(self.par_steps, X)]

    def inverse_transform(self, X):
        # NB: assumes each steps has an inverse transform
        return [step.inverse_transform(x) for step, x in zip(self.par_steps, X)]


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

            if not hasattr(step[1], "fit"):
                step = (step[0], TransformerAdapter(step[1]))

            adapted_steps.append(step)

        super().__init__(steps=adapted_steps)

    def __sklearn_clone__(self):
        return AdapterPipeline(steps=self._unadapted_steps)

    def predict_eval(self, X, y=None):
        check_is_fitted(self)

        Xt = X
        for _, name, transform in self._iter(with_final=False):
            if hasattr(transform, "transform_eval"):
                Xt = transform.transform_eval(Xt, y)
            else:
                Xt = transform.transform(Xt)

            if isinstance(Xt, tuple):
                Xt, _ = Xt

        last_step = self.steps[-1][1]
        if hasattr(last_step, "predict_eval"):
            return last_step.predict_eval(Xt, y)

        return last_step.predict(Xt)


class AdapterFeatureUnion(FeatureUnion):
    def __init__(
        self,
        transformer_list,
        *,
        n_jobs=None,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=True,
    ):
        self._unadapted_transformer_list = transformer_list

        adapted_transformer_list = []
        for index, transformer in enumerate(transformer_list):
            if not isinstance(transformer, Iterable):
                transformer_name = f"step_{index}"
                transformer = (transformer_name, transformer)

            if not hasattr(transformer[1], "fit"):
                transformer = (transformer[0], TransformerAdapter(transformer[1]))

            adapted_transformer_list.append(transformer)

        super().__init__(
            adapted_transformer_list,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights,
            verbose=verbose,
            verbose_feature_names_out=verbose_feature_names_out,
        )

    def __sklearn_clone__(self):
        return AdapterFeatureUnion(
            self._unadapted_transformer_list,
            n_jobs=self.n_jobs,
            transformer_weights=self.transformer_weights,
            verbose=self.verbose,
            verbose_feature_names_out=self.verbose_feature_names_out,
        )


class EvaluatedModel(BaseEstimator, TransformerMixin):
    """Model with evaluation.

    Wraps a model to log info.

    Parameters
    ----------
    model : sklearn.BaseEstimator
        Estimator being evaluated.
    evaluator : polpo.ModelEvaluator
        Model evaluator.
    """

    # TODO: need to think about inheritance
    # TODO: split in two?

    def __init__(self, model, evaluator):
        super().__init__()
        self.model = model
        self.evaluator = evaluator
        self.eval_result_ = None
        self.eval_result_pred_ = None

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped model."""
        return getattr(self.model, name)

    def __sklearn_clone__(self):
        return EvaluatedModel(model=clone(self.model), evaluator=self.evaluator)

    def fit(self, X, y=None):
        self.model = self.model.fit(X, y)

        self.eval_result_ = self.evaluator(self.model, X, y)
        return self

    def predict_eval(self, X, y=None):
        check_is_fitted(self)

        if hasattr(self.model, "predict_eval"):
            y_pred = self.model.predict_eval(X, y)
        else:
            y_pred = self.model.predict(X)

        self.eval_result_pred_ = self.evaluator(self.model, X, y, y_pred=y_pred)
        return y_pred

    def transform_eval(self, X, y=None):
        check_is_fitted(self)

        Xt = self.model.transform(X, y)
        yt = y
        # e.g. PLS
        if is_tuple := isinstance(Xt, tuple):
            Xt, yt = Xt

        # TODO: potentially pass Xt?
        self.eval_result_pred_ = self.evaluator(self.model, X, y)

        if is_tuple:
            return Xt, yt

        return Xt

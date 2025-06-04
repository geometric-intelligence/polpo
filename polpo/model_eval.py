import abc

import numpy as np
import scipy
from sklearn.metrics import r2_score
from statsmodels.stats.multitest import multipletests

from polpo.preprocessing import Map
from polpo.preprocessing.mesh.conversion import ToVertices
from polpo.sklearn.np import FlattenButFirst

# NB: this applies to sklearn models


class ModelEvaluator:
    def __init__(self, prefix="", extender=None):
        # TODO: rename extender
        if extender is None:
            extender = lambda x: x

        self.prefix = prefix
        self.extender = extender

    @abc.abstractmethod
    def __call__(self, model, X, y=None, y_pred=None):
        pass

    def _prefix_keys(self, results):
        if self.prefix:
            return dict(
                (f"{self.prefix}-{key}", value) for (key, value) in results.items()
            )

        return results

    def _update_results(self, results):
        return self._prefix_keys(self.extender(results))


class MultiEvaluator(ModelEvaluator):
    def __init__(self, evaluators, extender=None):
        super().__init__("", extender)
        self.evaluators = evaluators

    def __call__(self, model, X, y=None, y_pred=None):
        # assumes X does not contain intercept

        if hasattr(model, "predict") and y_pred is None:
            y_pred = model.predict(X)

        results = {}
        for evaluator in self.evaluators:
            results_ = evaluator(model, X, y, y_pred)
            results.update(results_)

        return self._update_results(results)


class PValuesAdjuster:
    # TODO: implement own version to avoid installing multipletests?
    # NB: some of them are trivial
    def __init__(self, method="bonferroni"):
        # TODO: add method validation; add also two-stage?
        self.method = method

    def __call__(self, pvals):
        return multipletests(pvals.flatten(), method="bonferroni")[1].reshape(
            pvals.shape
        )


class OlsPValues(ModelEvaluator):
    def __init__(self, adjust_pvalues=None, prefix="", extender=None):
        super().__init__(prefix, extender)
        if adjust_pvalues is None:
            adjust_pvalues = PValuesAdjuster()

        self.adjust_pvalues = adjust_pvalues

    def __call__(self, model, X, y, y_pred=None):
        # assumes X does not contain intercept

        if y_pred is None:
            y_pred = model.predict(X)

        y = y[:, None] if y.ndim == 1 else y
        y_pred = y_pred[:, None] if y_pred.ndim == 1 else y_pred

        mse = np.mean(((y_pred - y) ** 2), axis=0)

        n = y.shape[0]
        p = X.shape[1] + model.fit_intercept

        res_var = mse * n / (n - p)

        X_with_intercept = np.c_[np.ones((X.shape[0], 1)), X]

        cov = np.linalg.inv(X_with_intercept.transpose() @ X_with_intercept)

        # take into account fit_intercept
        index = int(model.fit_intercept)
        cov_beta = res_var[:, None] * (np.diag(cov)[index:])

        std_err = np.sqrt(cov_beta)

        t = model.coef_ / std_err

        df = n - p

        pvalues = 2 * (1 - scipy.stats.t.cdf(abs(t), df))

        res = {
            "mse": mse,
            "res_var": res_var,
            "std_err": std_err,
            "t": t,
            "pvals": pvalues,
        }

        if self.adjust_pvalues is not None:
            res["adj-pvals"] = self.adjust_pvalues(res["pvals"])

        return self._update_results(res)


class RegressionMetricAdapter(ModelEvaluator):
    # adapts sklearn metric
    def __init__(self, name, func, multioutput="raw_values", prefix="", extender=None):
        super().__init__(prefix, extender)
        self.name = name
        self.func = func
        self.multioutput = multioutput

    def __call__(self, model, X, y, y_pred=None):
        # model and X are ignored if y_pred is not None
        if y_pred is None:
            y_pred = model.predict(X)

        return self._update_results(
            {self.name: self.func(y, y_pred, multioutput=self.multioutput)}
        )


class R2Score(RegressionMetricAdapter):
    def __init__(self, multioutput="raw_values", prefix="", extender=None):
        super().__init__(
            "r2", r2_score, multioutput=multioutput, prefix=prefix, extender=extender
        )


class MeshR2Score(ModelEvaluator):
    def __init__(self, as_dict=True, prefix="", extender=None):
        super().__init__(prefix, extender=extender)
        self.as_dict = as_dict

    def __call__(self, model, X, y, y_pred=None):
        if y_pred is None:
            y_pred = model.predict(X)

        meshes2vertices = (
            Map(ToVertices()) + (lambda x: np.stack(x)) + FlattenButFirst()
        )

        vertices = meshes2vertices(y)
        vertices_pred = meshes2vertices(y_pred)

        featurewise_r2 = r2_score(vertices, vertices_pred, multioutput="raw_values")

        if self.as_dict:
            return self._update_results({"featurewise_r2": featurewise_r2})

        return featurewise_r2


class MeshEuclideanR2Score(ModelEvaluator):
    def __init__(self, as_dict=True, prefix="", extender=None):
        super().__init__(prefix, extender=extender)
        self.as_dict = as_dict

    def __call__(self, model, X, y, y_pred=None):
        if y_pred is None:
            y_pred = model.predict(X)

        meshes2vertices = Map(ToVertices()) + (lambda x: np.stack(x))

        vertices = meshes2vertices(y)
        vertices_pred = meshes2vertices(y_pred)

        vertices_mean = np.mean(vertices, axis=0)

        ss_res = np.sum(
            np.linalg.norm((vertices - vertices_pred), axis=-1) ** 2, axis=0
        )
        ss_tot = np.sum(
            np.linalg.norm((vertices - vertices_mean) ** 2, axis=-1), axis=0
        )

        vertexwise_r2 = 1 - ss_res / ss_tot

        if self.as_dict:
            return self._update_results({"vertexwise_r2": vertexwise_r2})

        return vertexwise_r2


class PcaEvaluator(ModelEvaluator):
    def __call__(self, model, X, y=None, y_pred=None):
        return self._update_results(
            {
                "expl_var": model.explained_variance_,
                "expl_var_ratio": model.explained_variance_ratio_,
                "expl_var_ratio-cum": np.cumsum(model.explained_variance_ratio_),
            }
        )


class ReconstructionError(ModelEvaluator):
    def __call__(self, model, X, y=None, y_pred=None):
        # NB: X is flattened at this point
        X_recon = model.inverse_transform(model.transform(X))

        diff2 = (X - X_recon) ** 2

        featurewise_rec_error = np.sum(diff2, axis=0)
        rec_error_sum = np.sum(diff2)
        rec_error_mse = np.mean(diff2)

        return self._update_results(
            {
                "featurewise_rec_error": featurewise_rec_error,
                "rec_error_sum": rec_error_sum,
                "rec_error_mse": rec_error_mse,
            }
        )


class VertexReconstructionError(ModelEvaluator):
    def __init__(self, dim=3, prefix=""):
        super().__init__(prefix)
        self.dim = dim

    def __call__(self, model, X, y=None, y_pred=None):
        # NB: X is flattened at this point
        X_recon = model.inverse_transform(model.transform(X))

        X = X.reshape(X.shape[0], -1, self.dim)
        X_recon = X_recon.reshape(X.shape[0], -1, self.dim)

        diff2 = np.linalg.norm(X - X_recon, axis=-1) ** 2

        vertexwise_rec_error = np.sum(diff2, axis=0)
        rec_error_sum = np.sum(diff2)
        rec_error_mse = np.mean(diff2)

        return self._update_results(
            {
                "vertexwise_rec_error": vertexwise_rec_error,
                "rec_error_sum": rec_error_sum,
                "rec_error_mse": rec_error_mse,
            }
        )


class ResultsExtender:
    def __init__(self, metrics=None):
        self.metrics = metrics

    def __call__(self, results):
        metrics = list(results.keys()) if self.metrics is None else self.metrics

        for metric in metrics:
            val = results[metric]
            results.update(
                {
                    f"{metric}-mean": np.mean(val),
                    f"{metric}-max": np.max(val),
                    f"{metric}-min": np.min(val),
                }
            )

        return results

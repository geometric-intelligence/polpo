"""Workflow for evaluating truncated mesh prediction models."""

import json
import traceback
from datetime import datetime, timezone

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from polpo.sklearn.model_selection import (
    assemble_predictions,
    cross_fit,
    predict_folds,
)
from polpo.sklearn.truncation import truncate
from polpo.time import Timer


class TruncatedCVEvaluator:
    """Evaluate prediction across truncations of a fitted model.

    Parameters
    ----------
    estimator : estimator
        Maximal estimator to fit in each cross-validation fold. The fitted
        estimator must support :func:`polpo.sklearn.truncation.truncate`.
    truncations : iterable of int
        Truncation levels to evaluate.
    metrics : dict
        Mapping from metric names to callables accepting two objects.
    cv : cross-validation splitter
        Cross-validation splitting strategy.
    prepare_data : callable
        Function mapping a mesh dataset to ``X, y, groups, keys``.
    results_dir : pathlib.Path
        Directory where evaluation outputs are written.
    n_jobs : int, optional
        Number of jobs used during cross-fitting.
    metadata : dict, optional
        Additional experiment metadata.
    """

    PROTOCOL_VERSION = "0.1.0"

    def __init__(
        self,
        estimator,
        truncations,
        metrics,
        prepare_data,
        cv=None,
        results_dir=None,
        n_jobs=None,
        metadata=None,
    ):
        if cv is None:
            cv = LeaveOneGroupOut()

        self.timer = Timer()

        self.estimator = estimator
        self.cv = cv
        self.truncations = list(truncations)
        self.metrics = metrics
        self.prepare_data = prepare_data
        self.results_dir = results_dir
        self.n_jobs = n_jobs
        self.metadata = metadata or {}

        self.reset()

    def reset(self):
        """Reset protocol state."""
        self.timer.reset()

        self.params_ = {
            "version": self.PROTOCOL_VERSION,
            "metadata": self.metadata,
            "truncations": self.truncations,
            "metrics": list(self.metrics),
            "estimator": repr(self.estimator),
            "cv": repr(self.cv),
            "n_jobs": self.n_jobs,
        }

        self.results_ = {
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    def fit(self, X, y, groups):
        """Fit the maximal estimator across cross-validation folds."""
        with self.timer("fit"):
            return cross_fit(
                self.estimator,
                X,
                y,
                groups=groups,
                cv=self.cv,
                n_jobs=self.n_jobs,
            )

    def predict(self, cross_fit_result, X, truncation):
        """Predict held-out meshes at a given truncation level."""
        estimators = [
            truncate(estimator, truncation) for estimator in cross_fit_result.estimators
        ]

        predictions = predict_folds(
            estimators,
            X,
            cross_fit_result.test_indices,
        )

        return assemble_predictions(
            predictions,
            cross_fit_result.test_indices,
        )

    def evaluate(self, cross_fit_result, X, y):
        """Evaluate held-out predictions across truncation levels."""
        distances = {
            name: np.empty((len(y), len(self.truncations))) for name in self.metrics
        }

        with self.timer("evaluation"):
            for truncation_idx, truncation in enumerate(self.truncations):
                predictions = self.predict(
                    cross_fit_result,
                    X,
                    truncation,
                )

                for name, metric in self.metrics.items():
                    distances[name][:, truncation_idx] = [
                        metric(y_true, y_pred) for y_true, y_pred in zip(y, predictions)
                    ]

        return distances

    def write(self):
        """Write protocol parameters, results, timings, and distances."""
        if self.results_dir is None:
            return

        self.results_dir.mkdir(parents=True, exist_ok=True)

        with open(self.results_dir / "params.json", "w") as file:
            json.dump(self.params_, file, indent=2)

        with open(self.results_dir / "results.json", "w") as file:
            json.dump(self.results_, file, indent=2)

        with open(self.results_dir / "time.json", "w") as file:
            json.dump(self.timer.as_dict(), file, indent=2)

        if hasattr(self, "distances_"):
            np.savez_compressed(
                self.results_dir / "distances.npz",
                **self.distances_,
            )

    def _record_failure(self, error):
        self.results_.update(
            {
                "status": "failed",
                "failed_stage": self.current_stage_,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            }
        )

    def run(self, dataset):
        """Run truncated cross-validated mesh prediction evaluation."""
        self.reset()

        self.results_["status"] = "running"

        try:
            with self.timer("run"):
                self.current_stage_ = "data_preparation"
                X, y, groups, keys = self.prepare_data(dataset)

                self.current_stage_ = "cross_fit"
                cross_fit_result = self.fit(X, y, groups)

                self.current_stage_ = "evaluation"
                distances = self.evaluate(
                    cross_fit_result,
                    X,
                    y,
                )

                self.current_stage_ = "completed"

        except Exception as error:
            self._record_failure(error)
            self.write()
            raise

        self.results_.update(
            {
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "keys": list(keys),
            }
        )

        self.cross_fit_result_ = cross_fit_result
        self.distances_ = distances
        self.keys_ = keys

        self.write()

        return self

"""Workflow for evaluating truncated mesh prediction models."""

import traceback
from datetime import datetime, timezone

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from polpo.io.json import save_json
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
        fit_diagnostics=None,
        cv=None,
        results_dir=None,
        n_jobs=None,
        metadata=None,
    ):
        if cv is None:
            cv = LeaveOneGroupOut()

        self.timer = Timer()

        self.estimator = estimator
        self.fit_diagnostics = fit_diagnostics
        self.cv = cv
        self.truncations = list(truncations)
        self.metrics = metrics
        self.prepare_data = prepare_data
        self.results_dir = results_dir
        self.n_jobs = n_jobs
        self.metadata = metadata or {}

    def _reset_state(self):
        self.status_ = "running"
        self.current_stage_ = None
        self.failed_stage_ = None
        self.error_ = None

        self.cross_fit_result_ = None
        self.fit_diagnostics_ = None
        self.held_out_groups_ = None
        self.distances_ = None
        self.keys_ = None

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
        if self.results_dir is None:
            return

        save_json(self.results_dir / "params.json", self.to_params())
        save_json(self.results_dir / "results.json", self.to_results())

        if self.fit_diagnostics_ is not None:
            save_json(
                self.results_dir / "fit_diagnostics.json",
                self.fit_diagnostics_,
            )

        if self.distances_ is not None:
            np.savez_compressed(
                self.results_dir / "distances.npz",
                **self.distances_,
            )

    def to_params(self):
        return {
            "version": self.PROTOCOL_VERSION,
            "truncations": self.truncations,
            "metrics": list(self.metrics),
            "estimator": repr(self.estimator),
            "cv": repr(self.cv),
            "n_jobs": self.n_jobs,
            "metadata": self.metadata,
        }

    def to_results(self):
        results = {
            "status": self.status_,
            "keys": list(self.keys_) if self.keys_ is not None else None,
            "held_out_groups": (
                list(self.held_out_groups_)
                if self.held_out_groups_ is not None
                else None
            ),
        }

        if self.status_ == "failed":
            results["failed_stage"] = self.failed_stage_
            results["error"] = self.error_

        return results

    def _record_failure(self, error):
        self.status_ = "failed"
        self.failed_stage_ = self.current_stage_
        self.error_ = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }

    def run(self, dataset):
        """Run truncated cross-validated mesh prediction evaluation."""
        self._reset_state()

        self.timer.start_run()

        try:
            with self.timer("run"):
                self.current_stage_ = "data_preparation"
                X, y, groups, self.keys_ = self.prepare_data(dataset)

                self.current_stage_ = "cross_fit"
                self.cross_fit_result_ = self.fit(X, y, groups)

                self.held_out_groups_ = [
                    np.unique(groups[test_indices]).item()
                    for test_indices in self.cross_fit_result_.test_indices
                ]

                if self.fit_diagnostics is not None:
                    self.fit_diagnostics_ = self.fit_diagnostics(self.cross_fit_result_)

                self.current_stage_ = "evaluation"
                self.distances_ = self.evaluate(
                    self.cross_fit_result_,
                    X,
                    y,
                )

                self.status_ = self.current_stage_ = "completed"

        except Exception as error:
            self._record_failure(error)
            raise

        finally:
            self.timer.stop_run()
            self.write()

        return self

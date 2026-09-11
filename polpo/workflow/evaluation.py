"""Workflows for model evaluation."""

from pathlib import Path

from polpo.sklearn.evaluation import TruncatedCVEvaluator
from polpo.workflow.task import TaskRunner, task


class TruncatedCVEvaluationRunner(TaskRunner):
    """Run and persist truncated cross-validation evaluation."""

    def __init__(
        self,
        prepare_inputs,
        results_dir,
        state_dir=None,
        **evaluation_kwargs,
    ):
        if state_dir is None:
            state_dir = Path(results_dir)

        super().__init__(state_dir)

        self.prepare_inputs = prepare_inputs
        self.results_dir = results_dir
        self.evaluation_kwargs = evaluation_kwargs

    @task
    def evaluate(self):
        """Run the evaluation and persist its result."""
        dataset, estimator, metrics = self.prepare_inputs()

        evaluator = TruncatedCVEvaluator(
            estimator=estimator,
            metrics=metrics,
            **self.evaluation_kwargs,
        ).fit(dataset)

        evaluator.result_.to_dir(self.results_dir)

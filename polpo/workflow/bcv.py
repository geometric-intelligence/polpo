from pathlib import Path

from polpo.bcv.mesh import GroupedMeshRankSelection
from polpo.workflow.task import TaskRunner, task


class GroupedMeshRankSelectionRunner(TaskRunner):
    def __init__(
        self,
        prepare_inputs,
        results_dir,
        state_dir=None,
        **selection_kwargs,
    ):
        if state_dir is None:
            state_dir = (
                Path(".rank_selection") if results_dir is None else Path(results_dir)
            )

        super().__init__(state_dir)

        self.prepare_inputs = prepare_inputs

        self.results_dir = results_dir
        self.selection_kwargs = selection_kwargs

    @classmethod
    def from_data(
        cls,
        mesh_faces,
        dataset,
        results_dir,
        state_dir=None,
        **selection_kwargs,
    ):
        return cls(
            prepare_inputs=lambda: (mesh_faces, dataset),
            results_dir=results_dir,
            state_dir=state_dir,
            **selection_kwargs,
        )

    @task
    def select_rank(self):
        mesh_faces, dataset = self.prepare_inputs()

        selection = GroupedMeshRankSelection(
            **self.selection_kwargs,
        ).fit(mesh_faces, dataset)

        selection.result_.to_dir(self.results_dir)

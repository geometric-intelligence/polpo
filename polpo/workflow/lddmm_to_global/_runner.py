from functools import partial

from polpo.workflow.lddmm_to_global import LddmmToGlobal
from polpo.workflow.task import TaskRunner


class LddmmToGlobalRunner(TaskRunner):
    """Run LDDMM-to-global workflows over a collection of items.

    Each item defines an independent ``LddmmToGlobal`` run. Inputs for a run
    are constructed by ``prepare_inputs``, allowing dataset-specific loading
    and preprocessing to remain separate from the workflow orchestration.

    Parameters
    ----------
    items : iterable
        Items identifying the independent runs. Each item is passed to
        ``prepare_inputs`` and is used as the task name and output directory
        name.
    prepare_inputs : callable
        Function called as ``prepare_inputs(item)``. It must return
        ``dataset, atlas_keys, known_correspondences, metadata``, where
        ``dataset`` is the nested dataset passed to ``LddmmToGlobal.run``,
        ``atlas_keys`` identifies the local atlas observations, and
        ``known_correspondences`` indicates whether mesh correspondences are
        known, ``metadata`` contains information to persist with the run.
    results_dir : path-like
        Root directory for the runs. Results for each item are written to a
        subdirectory named after that item.
    **protocol_kwargs
        Additional keyword arguments passed to ``LddmmToGlobal``.
    """

    def __init__(
        self,
        keys,
        prepare_inputs,
        results_dir,
        **protocol_kwargs,
    ):
        super().__init__(results_dir)

        self.keys = keys
        self.prepare_inputs = prepare_inputs
        self.protocol_kwargs = protocol_kwargs

    def tasks(self):
        return {key: partial(self._run, key) for key in self.keys}

    def _run(self, key):
        dataset, atlas_keys, known_correspondences, metadata = self.prepare_inputs(key)

        protocol = LddmmToGlobal(
            results_dir=self.results_dir / key,
            metadata=metadata,
            known_correspondences=known_correspondences,
            **self.protocol_kwargs,
        )

        protocol.run(
            dataset,
            atlas_keys=atlas_keys,
        )

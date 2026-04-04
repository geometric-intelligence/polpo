import os
import re

import polpo.preprocessing.dict as ppdict
from polpo.preprocessing import (
    BranchingPipeline,
    ExceptionToWarning,
)
from polpo.preprocessing.load.bids import FoldersSelector as BidsFoldersSelector
from polpo.preprocessing.path import ExpandUser, FileFinder
from polpo.preprocessing.str import DigitFinder, StartsWith

from .utils import MATERNAL_IDS


def _session_sorter(session_id):
    return (
        re.sub(r"\d+$", "", session_id),
        DigitFinder(index=-1)(session_id),
    )


def FoldersSelector(
    derivative,
    subject_subset=None,
    session_subset=None,
    remove_repeated=True,
):
    """Create pipeline to load maternal sessions folder names.

    Assumes the following folder structure:
    - <data_dir>
        - maternal_brain_project
        - maternal_brain_project_pilot

    For each of the projects folder assumes:
    - <project_folder>
        - derivatives
            - <derivative_1>
            - <derivative_2>

    For each of the tools folder assumes:
    - <tool>
        - <session-folder>
        - ...

    Parameters
    ----------
    derivative : str
        Derivative folder starting (e.g. "fsl_first", "fastsurfer-long").
    subject_subset : array-like
        Id of the subjects. If None, assumes all.
    session_subset : array-like
        Subset of sessions to load. If `None`, loads all.

    Returns
    -------
    pipe : Pipeline
        Pipeline to load maternal sessions folder names.
    """
    project_folder = "maternal_brain_project"

    if subject_subset is None:
        subject_subset = MATERNAL_IDS

    if (
        session_subset is not None
        and len(subject_subset) > 1
        and "01" in subject_subset
    ):
        raise ValueError("Can't filter sessions if pilot included")

    for subject_id in subject_subset:
        if subject_id not in MATERNAL_IDS:
            raise ValueError(
                f"Oops, `{subject_id}` is not available. Please, choose from: {','.join(MATERNAL_IDS)}"
            )

    if "01" in subject_subset and len(subject_subset) > 1:
        subject_subset = list(subject_subset)
        subject_subset.remove("01")

        subject_subsets = [["01"], subject_subset]

    else:
        subject_subsets = [subject_subset]

    def _prepend_data_dir(project_folder):
        return lambda x: os.path.join(x, project_folder, "derivatives")

    pipes = []
    for subject_subset in subject_subsets:
        pilot = True if "01" in subject_subset else False

        project_folder_ = project_folder
        if pilot:
            project_folder_ += "_pilot"

        pipe = (
            _prepend_data_dir(project_folder_)
            + ExpandUser()
            + FileFinder(rules=StartsWith(derivative))
            + BidsFoldersSelector(
                subject_subset,
                session_subset=session_subset,
                session_sorter=_session_sorter,
            )
        )

        if pilot and remove_repeated:
            # same session metadata as 26
            pipe += ppdict.DictMap(
                ExceptionToWarning(ppdict.RemoveKeys(keys=[27]), warn=False)
            )

        pipes.append(pipe)

    if len(pipes) == 1:
        return pipes[0]

    pipe = BranchingPipeline(pipes, merger=lambda x: x[0] | x[1])

    return pipe

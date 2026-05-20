from pathlib import Path

from polpo.bids import DerFolderSelector
from polpo.enigma.logjac import load_logjac_group

# TODO: expand this to enigma roi?
from polpo.preprocessing import BranchingPipeline

from .defaults import PILOT_PROJECT_FOLDER, PROJECT_FOLDER
from .path import _split_subject_subset


def LogJacLoader(
    subject_subset=None,
    session_subset=None,
    struct_subset=None,
):
    pilot_subset, subject_subset = _split_subject_subset(subject_subset=None)
    derivative = "enigma"

    pipes = []
    if len(pilot_subset):
        # TODO: need to remove repeated?
        pipe = (
            (lambda folder: Path(folder).expanduser() / PILOT_PROJECT_FOLDER)
            + DerFolderSelector(derivative)
            + (lambda path: path / "data" / "subjects_file_LogJacs.csv")
            + (
                lambda filename: load_logjac_group(
                    filename,
                    subject_subset=pilot_subset,
                    session_subset=session_subset,
                    struct_subset=struct_subset,
                )
            )
        )
        pipes.append(pipe)

    if len(subject_subset):
        pipe = (
            (lambda folder: Path(folder).expanduser() / PROJECT_FOLDER)
            + DerFolderSelector(derivative)
            + (lambda path: path / "data" / "subjects_file_LogJacs.csv")
            + (
                lambda filename: load_logjac_group(
                    filename,
                    subject_subset=subject_subset,
                    session_subset=session_subset,
                    struct_subset=struct_subset,
                )
            )
        )
        pipes.append(pipe)

    if len(pipes) == 1:
        return pipes[0]

    return BranchingPipeline(pipes, merger=lambda data: data[0] | data[1])

import os

import polpo.preprocessing.dict as ppdict
from polpo.preprocessing.load.bids import FoldersSelector as BidsFoldersSelector
from polpo.preprocessing.path import ExpandUser, FileFinder
from polpo.preprocessing.str import StartsWith


def FoldersSelector(
    derivative,
    subject_subset=None,
    session_subset=None,
    remove_missing_sessions=True,
):
    """Create pipeline to select derivative session folders.

    The pipeline takes a dataset root directory as input and returns
    a nested dictionary indexed by subject and session identifiers:

    ``output[subject_id][session_id] -> folder_path``

    Parameters
    ----------
    derivative : str
        Name of the derivative folder (e.g. ``"fsl_first"``,
        ``"fastsurfer-long"``).
    subject_subset : array-like
        Subject identifiers to select. If ``None``, all subjects are used.
    session_subset : array-like
        Session identifiers to select. If ``None``, all sessions are used.
    remove_missing_sessions : bool
        Whether to keep only subjects with exactly two sessions.

    Returns
    -------
    pipe : Pipeline
        Pipeline mapping a dataset root directory to a nested dictionary
        of derivative session folder paths indexed by subject and session.
    """
    pipe = (
        (lambda x: os.path.join(x, "derivatives"))
        + ExpandUser()
        + FileFinder(rules=StartsWith(derivative))
        + BidsFoldersSelector(subject_subset, session_subset=session_subset)
    )

    if remove_missing_sessions:
        pipe += ppdict.DictFilter(func=lambda x: len(x) == 2)

    return pipe

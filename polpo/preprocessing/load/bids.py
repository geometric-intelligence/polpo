import polpo.preprocessing.dict as ppdict
from polpo.preprocessing import (
    ContainsAll,
    FilteredGroupBy,
    Map,
)
from polpo.preprocessing.path import (
    ExpandUser,
    FileFinder,
    PathShortener,
)
from polpo.preprocessing.str import (
    RegexGroupFinder,
    TryToInt,
)

# https://bids.neuroimaging.io/index.html


def DerivativeFoldersSelector(
    subject_subset=None, session_subset=None, session_sorter=True
):
    """Create pipeline to load derivative mesh folders.

    Parameters
    ----------
    session_sorter : callable
        If ``True``, sorts using identity. If ``None``, skips sorting.
        ``session_id`` is input.

    Returns
    -------
    pipe : Pipeline
        Pipeline whose output is dict[dict[str, list[str]]].
        Key represents participant id, nested key represents session,
        and value the corresponding folder names.
    """
    # TODO: also for this for raw
    if session_sorter is True:
        session_sorter = lambda x: x

    folders_selector = ExpandUser() + FileFinder(
        rules=ContainsAll(["sub", "ses"]), as_list=True
    )

    path_to_subject_id = PathShortener() + RegexGroupFinder(r"sub-([A-Za-z0-9]+)")
    path_to_session_id = (
        PathShortener() + RegexGroupFinder(r"ses-([A-Za-z0-9]+)") + TryToInt()
    )

    folders_selector += FilteredGroupBy(path_to_subject_id, subset=subject_subset)

    folders_selector += ppdict.DictMap(
        ppdict.HashWithIncoming(
            key_step=Map(path_to_session_id),
            key_subset=session_subset,
            key_sorter=session_sorter,
        )
    )

    return folders_selector

import polpo.preprocessing.dict as ppdict
from polpo.neuroi.mesh import MeshDatasetLoader as DerMeshDatasetLoader
from polpo.preprocessing import Constant, pipe_to_func

from .defaults import DATA_DIR
from .path import FoldersSelector
from .tabular import get_key_to_birth_week, get_key_to_week


def MeshDatasetLoader(
    derivative,
    data_dir=None,
    subject_subset=None,
    session_subset=None,
    struct_subset=None,
    mesh_reader=False,
    index_session_by="id",
):
    """Create pipeline to load derivative meshes.

    The pipeline takes no input. It selects subject-session folders from
    ``data_dir`` and returns a nested dictionary:

    ``output[subject_id][session_id][struct_id] -> filename_or_mesh``

    Parameters
    ----------
    derivative : str
        Name of the derivative folder (e.g. ``"fsl_first"``,
        ``"fastsurfer-long"``).
    data_dir : str
        Dataset root directory.
    subject_subset : array-like
        Subject identifiers to select. If ``None``, all subjects are used.
    session_subset : array-like
        Session identifiers to select. If ``None``, all sessions are used.
    struct_subset : array-like
        Structure identifiers to select. If ``None``, all structures are used.
    mesh_reader : callable or bool
        Mesh reader applied to each selected mesh filename. If ``False``,
        filenames are returned instead of loaded meshes.
    index_session_by : {"id", "gest_week", "birth"}
        Strategy used to index sessions in the output.

        - ``"id"``: keep the original session identifiers.
        - ``"gest_week"``: replace session identifiers with gestational
        weeks.
        - ``"birth"``: replace session identifiers with gestational weeks
        relative to birth (birth week corresponds to 0).

    Returns
    -------
    pipe : Pipeline
        Pipeline returning a nested dictionary of mesh filenames or loaded
        meshes indexed by subject, session, and structure identifiers.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    if index_session_by not in ("id", "gest_week", "birth"):
        raise ValueError("Can't handle indexing by ``{index_session_by}``")

    folders_selector = Constant(data_dir) + FoldersSelector(
        subject_subset=subject_subset,
        session_subset=session_subset,
        derivative=derivative,
    )

    mesh_finder = DerMeshDatasetLoader(
        struct_subset, derivative, mesh_reader=mesh_reader
    )

    pipe = folders_selector + ppdict.NestedDictMap(mesh_finder)

    if index_session_by == "gest_week":
        keys_to_weeks = get_key_to_week(data_dir, subject_subset=subject_subset)
        pipe += ppdict.RenameNestedKeys(keys_to_weeks)

    elif index_session_by == "birth":
        keys_to_weeks = get_key_to_birth_week(data_dir, subject_subset=subject_subset)
        pipe += ppdict.RenameNestedKeys(keys_to_weeks)

    return pipe


load_meshes = pipe_to_func(MeshDatasetLoader)

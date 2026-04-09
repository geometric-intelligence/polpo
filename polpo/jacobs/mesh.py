import polpo.preprocessing.dict as ppdict
from polpo.neuroi.mesh import MeshDatasetLoader as DerMeshDatasetLoader
from polpo.preprocessing import Constant, pipe_to_func

from .path import FoldersSelector


def MeshDatasetLoader(
    derivative,
    data_dir="~/.herbrain/data/maternal",
    subject_subset=None,
    session_subset=None,
    struct_subset=None,
    mesh_reader=False,
):
    """Create pipeline to load maternal mesh filenames.

    Check out https://fsl.fmrib.ox.ac.uk/fsl/docs/#/structural/first.

    Parameters
    ----------
    derivative : str
        Derivative folder starting (e.g. "fsl_first", "fastsurfer-long").
    data_dir : str
        Directory where data is stored.
    subject_id : str
        Identification of the subject. If None, assumes pilot.
    struct_subset : str
        One of the following: 'Thal', 'Caud', 'Puta', 'Pall',
        'BrStem', 'Hipp', 'Amyg', 'Accu'.
        Suffixed with 'L_' or 'R_' (except 'BrStem').
    left : bool
        Whether to load left side. Not applicable to 'BrStem'.
    subset : array-like
        Subset of sessions to load. If `None`, loads all.

    Returns
    -------
    pipe : Pipeline
        Pipeline whose output is a nested dict whose keys are
        subject_id, session_id, struct_id.
        Values are filename or mesh.
    """
    folders_selector = Constant(data_dir) + FoldersSelector(
        subject_subset=subject_subset,
        session_subset=session_subset,
        derivative=derivative,
    )

    mesh_finder = DerMeshDatasetLoader(
        struct_subset, derivative, mesh_reader=mesh_reader
    )

    return folders_selector + ppdict.NestedDictMap(mesh_finder)


load_meshes = pipe_to_func(MeshDatasetLoader)

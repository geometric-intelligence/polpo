import polpo.preprocessing.dict as ppdict
import polpo.utils as putils
from polpo.enigma.naming import enigma_id_to_first_struct, first_struct_to_enigma_id
from polpo.freesurfer.mesh import FreeSurferReader
from polpo.fsl.validation import validate_structs
from polpo.preprocessing import Map
from polpo.preprocessing.path import (
    FileFinder,
    PathShortener,
)
from polpo.preprocessing.str import (
    DigitFinder,
    EndsWithAny,
)
from polpo.pyvista.conversion import PvFromData

from .naming import get_all_structs


def MeshReader():
    return FreeSurferReader() + PvFromData()


def MeshDatasetLoader(struct_subset=None, mesh_reader=False):
    # TODO: rename to MeshDatasetLoader
    # pipeline takes dirname
    if mesh_reader is None:
        mesh_reader = MeshReader()
    elif mesh_reader is False:
        mesh_reader = None

    if struct_subset is None:
        struct_subset = get_all_structs()

    # TODO: use enigma version
    validate_structs(struct_subset)

    enigma_indices = [
        f"_{first_struct_to_enigma_id(struct)}" for struct in struct_subset
    ]
    rules = EndsWithAny(enigma_indices)
    path_to_struct_id = (
        PathShortener() + DigitFinder(index=-1) + enigma_id_to_first_struct
    )

    return FileFinder(rules=rules, as_list=True) + ppdict.HashWithIncoming(
        key_step=Map(path_to_struct_id),
        step=Map(mesh_reader),
        key_sorter=putils.custom_order(struct_subset),
    )

import polpo.preprocessing.dict as ppdict
import polpo.utils as putils
from polpo.freesurfer import FreeSurferReader  # TODO: make optional?
from polpo.preprocessing import IdentityStep, Map
from polpo.preprocessing.path import (
    FileFinder,
    IsFileType,
    PathShortener,
)
from polpo.preprocessing.str import (
    ContainsAny,
    DigitFinder,
    EndsWithAny,
)
from polpo.pyvista.conversion import PvFromData
from polpo.pyvista.io import PvReader

from .naming import (
    enigma_id_to_first_struct,
    first_struct_to_enigma_id,
    get_all_first_structs,
)
from .validation import validate_first_struct


def tool_to_mesh_reader(tool):
    # TODO: define better output? i.e. why pv?
    # TODO: prevent importing pv?

    if tool.startswith("enigma"):
        return FreeSurferReader() + PvFromData()
    else:
        return PvReader()


def MeshLoader(struct_subset=None, derivative="fsl", as_mesh=False):
    # pipeline takes dirname

    if struct_subset is None:
        struct_subset = get_all_first_structs(
            include_brstem=not derivative.startswith("enigma")
        )

    for struct in struct_subset:
        validate_first_struct(struct)

    if derivative.startswith("enigma"):
        enigma_indices = [
            f"_{first_struct_to_enigma_id(struct)}" for struct in struct_subset
        ]
        rules = EndsWithAny(enigma_indices)
        path_to_struct_id = (
            PathShortener() + DigitFinder(index=-1) + enigma_id_to_first_struct
        )

    else:
        rules = [IsFileType("vtk"), ContainsAny(struct_subset)]
        # e.g. sub-01_ses-01-L_Hipp_first.vtk
        path_to_struct_id = PathShortener() + (
            lambda x: x.split("-")[-1].split(".")[0][:-6]
        )

    if as_mesh:
        mesh_reader = tool_to_mesh_reader(derivative)
    else:
        mesh_reader = IdentityStep()

    return FileFinder(rules=rules, as_list=True) + ppdict.HashWithIncoming(
        key_step=Map(path_to_struct_id),
        step=Map(mesh_reader),
        key_sorter=putils.custom_order(struct_subset),
    )

import polpo.preprocessing.dict as ppdict
import polpo.utils as putils
from polpo.preprocessing import IdentityStep, Map
from polpo.preprocessing.mesh.conversion import PvFromData
from polpo.preprocessing.mesh.io import FreeSurferReader, PvReader
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

# TODO: load enigma extra stuff too

FIRST_STRUCTS = {
    "Thal",
    "Caud",
    "Puta",
    "Pall",
    "BrStem",
    "Hipp",
    "Amyg",
    "Accu",
}


ENIGMA_STRUCT2FIRST = {
    "L_Thal": 10,
    "L_Caud": 11,
    "L_Puta": 12,
    "L_Pall": 13,
    "L_Hipp": 17,
    "L_Amyg": 18,
    "L_Accu": 26,
    "R_Thal": 49,
    "R_Caud": 50,
    "R_Puta": 51,
    "R_Pall": 52,
    "R_Hipp": 53,
    "R_Amyg": 54,
    "R_Accu": 58,
}

FIRST2ENIGMA_STRUCT = {value: key for key, value in ENIGMA_STRUCT2FIRST.items()}


def tool_to_mesh_reader(tool):
    # TODO: define better output? i.e. why pv?
    # TODO: prevent importing pv?

    if tool.startswith("enigma"):
        return FreeSurferReader() + PvFromData()
    else:
        return PvReader()


def validate_first_struct(struct):
    if "_" in struct:
        struct = struct.split("_")[1]

    if struct not in FIRST_STRUCTS:
        raise ValueError(
            f"Oops, `{struct}` is not available. Please, choose from: {','.join(FIRST_STRUCTS)}"
        )


def first_struct_to_enigma_id(struct):
    return ENIGMA_STRUCT2FIRST[struct]


def enigma_id_to_first_struct(struct):
    return FIRST2ENIGMA_STRUCT[struct]


def get_all_first_structs(prefixed=True, include_brstem=True, order=False):
    if not prefixed:
        return FIRST_STRUCTS

    out = []
    for prefix in ("L", "R"):
        for struct in FIRST_STRUCTS:
            if struct == "BrStem":
                continue

            out.append(f"{prefix}_{struct}")

    if include_brstem:
        out.append("BrStem")

    if order:
        return out

    return out


def MeshLoader(struct_subset=None, derivative="fsl", as_mesh=False):
    # pipeline takes dirname

    if struct_subset is None:
        struct_subset = get_all_first_structs()

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

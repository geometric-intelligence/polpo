from polpo.preprocessing.mesh.conversion import PvFromData

# TODO: prevent this import?
from polpo.preprocessing.mesh.io import FreeSurferReader

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

ENIGMA_STRUCT2ID = {
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


def tool_to_mesh_reader(tool):
    # TODO: define better output? i.e. why pv?

    # update for other tools
    if tool.startswith("enigma"):
        return FreeSurferReader() + PvFromData()
    else:
        raise ValueError(f"Oops, don't know how to handle: {tool}")


def validate_first_struct(struct):
    if struct not in FIRST_STRUCTS:
        raise ValueError(
            f"Oops, `{struct}` is not available. Please, choose from: {','.join(FIRST_STRUCTS)}"
        )


def first_struct_to_enigma_id(struct):
    return ENIGMA_STRUCT2ID[struct]

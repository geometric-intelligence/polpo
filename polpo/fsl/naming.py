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

FIRST_STRUCTS_LONG = {
    "L_Thal": "Left Thalamus",
    "L_Caud": "Left Caudate",
    "L_Puta": "Left Putamen",
    "L_Pall": "Left Pallidum",
    "L_Hipp": "Left Hippocampus",
    "L_Amyg": "Left Amygdala",
    "L_Accu": "Left Accumbens",
    "R_Thal": "Right Thalamus",
    "R_Caud": "Right Caudate",
    "R_Puta": "Right Putamen",
    "R_Pall": "Right Pallidum",
    "R_Hipp": "Right Hippocampus",
    "R_Amyg": "Right Amygdala",
    "R_Accu": "Right Accumbens",
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


def first_struct_to_enigma_id(struct):
    return ENIGMA_STRUCT2FIRST[struct]


def get_first_struct_long_name(struct):
    return FIRST_STRUCTS_LONG[struct]


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
        return sorted(out)

    return out

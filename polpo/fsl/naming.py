# TODO: move to neuroi.subcortical.naming?

FIRST_STRUCTS = {  # https://fsl.fmrib.ox.ac.uk/fsl/docs/structural/first.html
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


def get_struct_long_name(struct):
    return FIRST_STRUCTS_LONG[struct]


def get_all_structs(prefixed=True, include_brstem=True, order=False):
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

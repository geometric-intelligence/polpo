SUBCORTICAL_STRUCTS = {
    # adopting fsl: https://fsl.fmrib.ox.ac.uk/fsl/docs/structural/first.html
    "Thal",
    "Caud",
    "Puta",
    "Pall",
    "BrStem",
    "Hipp",
    "Amyg",
    "Accu",
}
SUBCORTICAL_STRUCTS_LONG = {
    "BrStem": "Brain Stem",
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

SUBCORTICAL_NAME_TO_COLOR = {
    "BrStem": (119, 159, 176, 0),
    "L_Thal": (0, 118, 14, 0),
    "R_Thal": (0, 118, 14, 0),
    "L_Caud": (122, 186, 220, 0),
    "R_Caud": (122, 186, 220, 0),
    "L_Puta": (236, 13, 176, 0),
    "R_Puta": (236, 13, 176, 0),
    "L_Pall": (12, 48, 255, 0),
    "R_Pall": (12, 48, 255, 0),
    "L_Hipp": (220, 216, 20, 0),
    "R_Hipp": (220, 216, 20, 0),
    "L_Amyg": (103, 255, 255, 0),
    "R_Amyg": (103, 255, 255, 0),
    "L_Accu": (255, 165, 0, 0),
    "R_Accu": (255, 165, 0, 0),
}


def get_subcortical_struct_long_name(struct):
    return SUBCORTICAL_STRUCTS_LONG[struct]


def _get_all_subcortical_structs(
    structs, prefixed=True, only_bilateral=False, order=False
):
    if not prefixed:
        return structs

    out = []
    for prefix in ("L", "R"):
        for struct in structs:
            if struct == "BrStem":
                if not only_bilateral:
                    out.append(struct)

                continue

            out.append(f"{prefix}_{struct}")

    if order:
        return sorted(out)

    return out


def get_all_subcortical_structs(prefixed=True, only_bilateral=False, order=False):
    return _get_all_subcortical_structs(
        SUBCORTICAL_STRUCTS,
        prefixed=prefixed,
        only_bilateral=only_bilateral,
        order=order,
    )

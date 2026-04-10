from polpo.neuroi.naming import _get_all_subcortical_structs

SUBCORTICAL_STRUCTS = {
    # https://fsl.fmrib.ox.ac.uk/fsl/docs/structural/first.html
    # same as in neuroi.naming, but keeping it independent
    "Thal",
    "Caud",
    "Puta",
    "Pall",
    "BrStem",
    "Hipp",
    "Amyg",
    "Accu",
}


def get_all_subcortical_structs(prefixed=True, only_bilateral=False, order=False):
    return _get_all_subcortical_structs(
        SUBCORTICAL_STRUCTS,
        prefixed=prefixed,
        only_bilateral=only_bilateral,
        order=order,
    )

from polpo.fsl.naming import get_all_structs as get_all_first_structs

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


def enigma_id_to_first_struct(struct):
    return FIRST2ENIGMA_STRUCT[struct]


def get_all_structs(prefixed=True, order=False):
    return get_all_first_structs(include_brstem=False, prefixed=prefixed, order=order)

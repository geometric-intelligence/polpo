from .naming import FIRST_STRUCTS


def validate_struct(struct):
    # TODO: make general function
    if "_" in struct:
        struct = struct.split("_")[1]

    if struct not in FIRST_STRUCTS:
        raise ValueError(
            f"Oops, `{struct}` is not available. Please, choose from: {','.join(FIRST_STRUCTS)}"
        )


def validate_structs(structs):
    for struct in structs:
        validate_struct(struct)

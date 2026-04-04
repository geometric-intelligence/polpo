from .naming import FIRST_STRUCTS


def validate_first_struct(struct):
    if "_" in struct:
        struct = struct.split("_")[1]

    if struct not in FIRST_STRUCTS:
        raise ValueError(
            f"Oops, `{struct}` is not available. Please, choose from: {','.join(FIRST_STRUCTS)}"
        )

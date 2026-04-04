from polpo.fsl.mesh import MeshLoader, tool_to_mesh_reader  # noqa: F401
from polpo.fsl.mri import SegmentationsLoader  # noqa: F401
from polpo.fsl.naming import (  # noqa: F401
    ENIGMA_STRUCT2FIRST,
    FIRST2ENIGMA_STRUCT,
    FIRST_STRUCTS,
    FIRST_STRUCTS_LONG,
    enigma_id_to_first_struct,
    first_struct_to_enigma_id,
    get_all_first_structs,
    get_first_struct_long_name,
)
from polpo.fsl.validation import validate_first_struct  # noqa: F401

# TODO: load enigma extra stuff too

import logging

import polpo.preprocessing.dict as ppdict
from polpo.dataset import Dataset, NestedDataset
from polpo.jacobs.mesh import MeshDatasetLoader
from polpo.utils import NestedKeyCodec


def prepare_inputs(
    struct,
    subject_ids,
    data_dir,
    derivative="enigma",
):
    """Prepare Jacobs data for an LDDMM-to-global run.

    Load meshes for a single subcortical structure, select the observations
    used as local atlases, remove subjects without a suitable atlas, and
    encode dataset keys for use in output paths.

    Parameters
    ----------
    struct : str
        Subcortical structure to process.
    subject_ids : array-like
        Subject identifiers to include.
    data_dir : path-like
        Root directory containing the Jacobs dataset.
    derivative : str
        Mesh derivative to load.

    Returns
    -------
    dataset : NestedDataset
        Mesh dataset with encoded keys.
    atlas_keys : NestedDataset
        Keys identifying the observations used as local atlases.
    metadata : dict
        Metadata describing the prepared dataset, including the key mapping
        and atlas keys.
    known_correspondences : bool
        Whether the loaded meshes have known vertex correspondences.
    """
    metadata = dict(
        struct=struct,
        subject_ids=subject_ids,
        derivative=derivative,
        data_dir=data_dir,
    )

    dataset = (
        MeshDatasetLoader(
            data_dir=data_dir,
            subject_subset=subject_ids,
            struct_subset=[struct],
            derivative=derivative,
            mesh_reader=None,
        )
        + ppdict.ExtractUniqueKey(nested=True)
        + NestedDataset
    )()

    last_keys = Dataset(dataset.inner_keys()).map_values(lambda x: x[-1])

    def local_template_filter(subj, session):
        control = int(subj[0]) > 1
        if control:
            return True
        else:
            return last_keys[subj] == session

    # ignores subjects with no atlas keys
    atlas_keys = dataset.filter_keys(local_template_filter)

    missing_pre = set(dataset.keys_list()) - set(atlas_keys.keys_list())
    if missing_pre:
        logging.info(
            f"Dropping subjects {missing_pre} because they do not have baseline meshes"
        )
    dataset = dataset.drop_outer(missing_pre)

    # encode dataset keys for manageable folder names
    key_codec = NestedKeyCodec.from_dataset(dataset)

    metadata["key_map"] = key_codec.to_dict()
    mapped_atlas_keys = metadata["atlas_keys"] = key_codec.encode_nested_keys(
        atlas_keys.inner_keys()
    )

    known_correspondences = True if derivative == "enigma" else False
    return (
        key_codec.encode_dataset(dataset),
        mapped_atlas_keys,
        known_correspondences,
        metadata,
    )

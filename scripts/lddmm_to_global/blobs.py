import shutil
import string
from pathlib import Path

import polpo.utils as putils
from polpo.lddmm_to_global import LddmmToGlobal
from polpo.surface_mesh.generation.blob import create_blob
from polpo.utils import NestedKeyCodec

if __name__ == "__main__":
    outputs_dir = putils.get_results_path() / "blobs/lddmm_to_global"

    if outputs_dir.exists():
        shutil.rmtree(outputs_dir)

    outputs_dir.mkdir(parents=True, exist_ok=False)

    dataset = {}
    for subj_index, (n_meshes, bump_amp, n_bumps) in enumerate(
        zip((3, 2, 4), (0.2, 0.3, 0.4), (3, 5, 6))
    ):
        dataset[string.ascii_uppercase[subj_index + 3]] = {
            index + 2: create_blob(
                resolution=10, bump_amp=bump_amp, n_bumps=n_bumps, smoothing_iter=10
            )
            for index in range(n_meshes)
        }

    atlas_keys = {
        "D": [2, 3],
        "E": [2],
        "F": [2, 3],
    }

    key_codec = NestedKeyCodec.from_dataset(dataset)

    mapped_atlas_keys = key_codec.encode_nested_keys(atlas_keys)

    params = {"key_map": key_codec.to_dict(), "atlas_keys": mapped_atlas_keys}
    protocol = LddmmToGlobal(
        known_correspondences=True,
        results_dir=outputs_dir,
        params=params,
    )

    protocol.run(key_codec.encode_dataset(dataset), atlas_keys=mapped_atlas_keys)

    shutil.copy(Path(__file__).resolve(), outputs_dir)

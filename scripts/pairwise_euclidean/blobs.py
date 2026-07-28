import shutil
import string

import polpo.utils as putils
from polpo.pairwise_dataset.euclidean import PairwiseEuclidean
from polpo.surface_mesh.generation.blob import create_blob

if __name__ == "__main__":
    outputs_dir = putils.get_results_path() / "blobs/pairwise_euclidean"

    if outputs_dir.exists():
        shutil.rmtree(outputs_dir)

    outputs_dir.mkdir(parents=True, exist_ok=False)

    # TODO: make a function out of it?
    dataset = {}
    for subj_index, (n_meshes, bump_amp, n_bumps) in enumerate(
        zip((3, 2), (0.2, 0.3), (3, 5))
    ):
        dataset[string.ascii_uppercase[subj_index]] = {
            index: create_blob(
                resolution=10, bump_amp=bump_amp, n_bumps=n_bumps, smoothing_iter=10
            )
            for index in range(n_meshes)
        }

    protocol = PairwiseEuclidean(
        results_dir=outputs_dir,
        n_jobs=1,
    )

    protocol.run(dataset)

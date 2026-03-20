if __name__ == "__main__":
    import shutil
    import string

    import polpo.utils as putils
    from polpo.mesh.generation.blob import create_blob
    from polpo.protocol.lddmm_to_global import LddmmToGlobal

    outputs_dir = putils.get_results_path() / "lddmm_to_global/debug/blobs"

    if outputs_dir.exists():
        shutil.rmtree(outputs_dir)

    outputs_dir.mkdir(parents=True, exist_ok=False)

    nested_dataset = {}
    for subj_index, (n_meshes, bump_amp, n_bumps) in enumerate(
        zip((3, 2, 4), (0.2, 0.3, 0.4), (3, 5, 6))
    ):
        nested_dataset[string.ascii_uppercase[subj_index + 3]] = {
            index + 2: create_blob(
                resolution=10, bump_amp=bump_amp, n_bumps=n_bumps, smoothing_iter=10
            )
            for index in range(n_meshes)
        }

    atlases_keys = {
        "D": [2, 3],
        "E": [2],
        "F": [2, 3],
    }

    protocol = LddmmToGlobal(
        known_correspondences=True,
        results_dir=outputs_dir,
    )

    protocol.run(nested_dataset, atlases_keys=atlases_keys)

if __name__ == "__main__":
    import shutil
    import string

    import polpo.utils as putils
    from polpo.mesh.generation.blob import create_blob
    from polpo.protocol.pairwise_varifold import PairwiseVarifold

    outputs_dir = putils.get_results_path() / "pairwise_varifold/blobs"

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

    protocol = PairwiseVarifold(
        known_correspondences=True,
        results_dir=outputs_dir,
        n_jobs=1,
        backend="keops",
    )

    protocol.run(dataset)

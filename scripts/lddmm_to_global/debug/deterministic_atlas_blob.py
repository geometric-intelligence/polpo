if __name__ == "__main__":
    import shutil
    import string

    import polpo.utils as putils
    from polpo.mesh.deformetrica import FrechetMean, LddmmMetric, Point
    from polpo.mesh.generation.blob import create_blob
    from polpo.preprocessing.mesh.registration import RigidAlignment

    bump_amp = 0.2
    outputs_dir = (
        putils.get_results_path() / "lddmm_to_global/debug/blob_deterministic_atlas"
    )

    if outputs_dir.exists():
        shutil.rmtree(outputs_dir)

    outputs_dir.mkdir(parents=True, exist_ok=False)

    n_meshes = 3
    bump_amp = 0.2

    raw_meshes = [
        create_blob(resolution=10, bump_amp=bump_amp, n_bumps=5, smoothing_iter=10)
        for _ in range(n_meshes)
    ]

    prep_pipe = RigidAlignment(known_correspondences=True)

    meshes = prep_pipe(raw_meshes)

    kernel_width = 2 * bump_amp
    registration_kwargs = dict(
        kernel_width=kernel_width,
        regularisation=1.0,
        max_iter=2000,
        freeze_control_points=False,
        metric="varifold",
        tol=1e-16,
        attachment_kernel_width=bump_amp,
    )

    metric = LddmmMetric(outputs_dir, **registration_kwargs)

    dataset = [
        Point(
            id_=string.ascii_uppercase[index],
            pv_surface=mesh,
            dirname=metric.meshes_dir,
        )
        for index, mesh in enumerate(meshes)
    ]

    estimator = FrechetMean(
        metric,
        initial_step_size=1e-1,
    )

    estimator.fit(dataset, atlas_id="atlas")

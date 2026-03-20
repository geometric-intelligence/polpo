# to check how long it takes to create a deterministic atlas with and without decimation

if __name__ == "__main__":
    import json
    from pathlib import Path

    import polpo.preprocessing.dict as ppdict
    import polpo.utils as putils
    from polpo.mesh.deformetrica import FrechetMean, LddmmMetric, Point
    from polpo.mesh.surface import PvSurface
    from polpo.mesh.varifold.tuning.geometry_based import SigmaFromLengths
    from polpo.preprocessing.load.pregnancy.jacobs import MeshLoader, get_key_to_week
    from polpo.preprocessing.mesh.registration import RigidAlignment
    from polpo.time import Timer

    timer = Timer()

    struct = "L_Hipp"
    subject_id = "01"
    derivative = "enigma"

    ratio_kernel = 1.2

    outputs_dir = (
        putils.get_results_path()
        / "lddmm_to_global"
        / f"deterministic_atlas_{subject_id}_{struct}_{derivative}"
    )
    outputs_dir.mkdir(parents=True, exist_ok=False)

    known_correspondences = True if derivative == "enigma" else False

    data_dir = (
        "/scratch/data/maternal"
        if putils.in_frank()
        else Path.home() / ".herbrain/data/maternal"
    )

    raw_meshes = (  # session_id
        MeshLoader(
            data_dir=data_dir,
            subject_subset=[subject_id],
            struct_subset=[struct],
            derivative=derivative,
            as_mesh=True,
        )
        + ppdict.ExtractUniqueKey(nested=True)
    )()

    key2week = get_key_to_week(data_dir=data_dir)[subject_id]
    key_filter = ppdict.DictFilter(func=(lambda x: x < 0))
    pre_keys = list(key_filter(key2week).keys())
    filt_meshes = ppdict.SelectKeySubset(pre_keys)(raw_meshes)

    if len(filt_meshes) < 1:
        raise ValueError()

    # preprocess
    align_pipe = RigidAlignment(
        target=putils.get_first(filt_meshes),
        known_correspondences=known_correspondences,
    )
    prep_pipe = align_pipe + ppdict.DictMap(PvSurface)
    meshes = prep_pipe(filt_meshes)

    # tune kernels
    sigma_search = SigmaFromLengths(
        ratio_charlen_mesh=2.0,
        ratio_charlen=0.25,
    )
    sigma_search.fit([putils.get_first(meshes)])

    sigma_var = sigma_search.sigma_
    sigma_vel = ratio_kernel * sigma_var

    registration_kwargs = dict(
        kernel_width=sigma_vel,
        regularisation=1.0,
        max_iter=2000,
        freeze_control_points=False,
        metric="varifold",
        tol=1e-16,
        attachment_kernel_width=sigma_var,
    )

    metric = LddmmMetric(outputs_dir, **registration_kwargs)

    dataset = [
        Point(
            id_=key,
            pv_surface=mesh,
            dirname=metric.meshes_dir,
        )
        for key, mesh in meshes.items()
    ]

    estimator = FrechetMean(
        metric,
        initial_step_size=1e-1,
    )

    timer = Timer()

    with timer("run"):
        estimator.fit(dataset, atlas_id="atlas")

    with open(outputs_dir / "time.json", "w") as file:
        json.dump(timer.as_dict(), file, indent=4)

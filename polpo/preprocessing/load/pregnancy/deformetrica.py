import random

import polpo.preprocessing.dict as ppdict
from polpo.mesh.surface import PvSurface
from polpo.preprocessing import CachablePipeline, Map
from polpo.preprocessing.load.pregnancy.jacobs import MeshLoader, get_subject_ids
from polpo.preprocessing.mesh.decimation import PvDecimate
from polpo.preprocessing.mesh.io import PvReader, PvWriter
from polpo.preprocessing.mesh.registration import RigidAlignment


class TwoRandomMaternalMeshesPipe(CachablePipeline):
    def __init__(
        self,
        outputs_dir,
        mesh_names=("mesh_a", "mesh_b"),
        struct_name="L_Hipp",
        target_reduction=0.6,
        as_pv_surface=True,
    ):
        subject_ids = random.sample(get_subject_ids(include_male=False, sort=True), 2)

        meshes_writer = (
            lambda dir_and_meshes: [
                (f"{dir_and_meshes[0]}/{name}", mesh)
                for mesh, name in zip(dir_and_meshes[1], mesh_names)
            ]
        ) + Map(PvWriter(ext="vtk"))

        vtk_loader = ppdict.HashWithIncoming(
            step=Map(PvReader() + (PvSurface if as_pv_surface else None)),
        )

        cache_loader = (
            lambda outputs_dir: [outputs_dir / f"{name}.vtk" for name in mesh_names]
        ) + vtk_loader

        pregnancy_loader = (
            (
                MeshLoader(
                    subject_subset=subject_ids,
                    struct_subset=[struct_name],
                    session_subset=None,
                    derivative="enigma",
                    as_mesh=True,
                )
                # TODO: split here when getting new dataset
                + ppdict.DictMap(ppdict.ExtractRandomKey())
                + ppdict.ExtractUniqueKey(nested=True)
                + ppdict.DictToValuesList()
                + RigidAlignment(
                    target=lambda x: x[0],
                    known_correspondences=True,
                )
                + Map(
                    PvDecimate(
                        target_reduction=target_reduction, volume_preservation=True
                    )
                    if target_reduction
                    else None
                )
            )
            + (lambda data: (outputs_dir, data))
            + meshes_writer
            + vtk_loader
        )

        super().__init__(
            cache_dir=outputs_dir,
            no_cache_pipe=pregnancy_loader,
            cache_pipe=cache_loader,
            to_cache_pipe=lambda x: x,  # TODO: handle this better
            use_cache=True,
        )


def get_two_random_maternal_meshes(
    outputs_dir,
    mesh_names=("mesh_a", "mesh_b"),
    struct_name="L_Hipp",
    target_reduction=0.6,
    as_pv_surface=True,
):
    pipe = TwoRandomMaternalMeshesPipe(
        outputs_dir,
        mesh_names=mesh_names,
        struct_name=struct_name,
        target_reduction=target_reduction,
        as_pv_surface=as_pv_surface,
    )
    return pipe()

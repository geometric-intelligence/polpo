import random

import polpo.preprocessing.dict as ppdict
from polpo.mesh.surface import PvSurface
from polpo.preprocessing import Map, Pipeline
from polpo.preprocessing.load.pregnancy.jacobs import MeshLoader, get_subject_ids
from polpo.preprocessing.mesh.decimation import PvDecimate
from polpo.preprocessing.mesh.registration import RigidAlignment


class TwoRandomMeshesPipe(Pipeline):
    def __init__(
        self,
        struct_name="L_Hipp",
        target_reduction=0.6,
        align=True,
        same_subject=False,
        as_pv_surface=False,
    ):
        # TODO: add possibility of loading carmona meshes?

        # TODO: update to use same_subject
        subject_ids = random.sample(get_subject_ids(include_male=False, sort=True), 2)

        pipe = (
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
            + (
                RigidAlignment(
                    target=lambda x: x[0],
                    known_correspondences=True,
                )
                if align
                else None
            )
            + Map(
                PvDecimate(target_reduction=target_reduction, volume_preservation=True)
                if target_reduction
                else None
            )
            + (Map(PvSurface) if as_pv_surface else None)
        )

        super().__init__(steps=pipe.steps)

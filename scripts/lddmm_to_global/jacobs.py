import json
import logging
from pathlib import Path

import polpo.preprocessing.dict as ppdict
import polpo.utils as putils
from polpo.enigma.naming import get_all_structs
from polpo.jacobs.mesh import MeshDatasetLoader
from polpo.jacobs.tabular import get_key_to_week
from polpo.jacobs.utils import get_subject_ids
from polpo.protocol.lddmm_to_global import LddmmToGlobal


def protocol_per_struct(
    struct,
    subject_ids,
    data_dir,
    results_dir,
    atlases_keys,
    derivative="enigma",
    ratio_charlen=0.25,
    ratio_kernel=1.5,
):
    # for serialization
    params = dict(
        struct=struct,
        subject_ids=subject_ids,
        derivative=derivative,
        data_dir=data_dir,
    )
    with open(results_dir / "params_maternal.json", "w") as file:
        json.dump(params, file, indent=4)

    known_correspondences = True if derivative == "enigma" else False

    dataset = (
        MeshDatasetLoader(
            data_dir=data_dir,
            subject_subset=subject_ids,
            struct_subset=[struct],
            derivative=derivative,
            as_mesh=True,
        )
        + ppdict.ExtractUniqueKey(nested=True)
    )()

    protocol = LddmmToGlobal(
        known_correspondences,
        results_dir=results_dir,
    )

    protocol.run(dataset, atlases_keys=atlases_keys)


if __name__ == "__main__":
    logging.basicConfig(
        filename="lddmm_to_global_jacobs.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

    # structs = get_all_structs(order=True)
    structs = ["L_Hipp", "R_Hipp", "L_Puta", "R_Puta"]

    subject_ids = get_subject_ids(sort=True)
    subject_ids.remove("1009B")  # no pre meshes

    derivative = "enigma"

    data_dir = (
        "/scratch/data/maternal"
        if putils.in_frank()
        else Path.home() / ".herbrain/data/maternal"
    )

    key2week = get_key_to_week(data_dir=data_dir)
    key_filter = ppdict.DictFilter(func=(lambda x: x < 0))
    atlases_keys = {}
    for subject_id in subject_ids:
        atlases_keys[subject_id] = keys = list(key_filter(key2week[subject_id]).keys())
        if len(keys) < 1:
            raise ValueError(f"No pre meshes for {subject_id}")

    for struct in structs:
        results_dir = (
            putils.get_results_path()
            / "lddmm_to_global"
            / "jacobs"
            / f"{struct}_{derivative}"
        )
        if results_dir.exists():
            logging.info(f"Skipping {struct} because folder already exists")
            continue

        results_dir.mkdir(parents=True, exist_ok=False)

        try:
            protocol_per_struct(
                struct,
                subject_ids=subject_ids,
                data_dir=data_dir,
                results_dir=results_dir,
                atlases_keys=atlases_keys,
                ratio_charlen=0.25,
                ratio_kernel=1.5,
            )
        except Exception as e:
            logging.warning(f"Oops, something went wrong for {struct}: {e}")

    logging.info("Done")

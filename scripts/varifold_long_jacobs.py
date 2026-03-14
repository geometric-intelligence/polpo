import json
import logging
from pathlib import Path

import polpo.preprocessing.dict as ppdict
import polpo.utils as putils
from polpo.preprocessing.load.fsl import get_all_first_structs
from polpo.preprocessing.load.pregnancy.jacobs import (
    MeshLoader,
    get_subject_ids,
)
from polpo.protocol.varifold import PairwiseVarifold


def protocol_per_struct(
    struct, subject_ids, data_dir, results_dir, derivative="enigma", subsample=None
):
    # for serialization
    params = dict(
        struct=struct,
        subject_ids=subject_ids,
        derivative=derivative,
        data_dir=data_dir,
    )
    with open(results_dir / "params.json", "w") as file:
        json.dump(params, file, indent=4)

    known_correspondences = True if derivative == "enigma" else False

    mesh_loader = (
        MeshLoader(
            data_dir=data_dir,
            subject_subset=subject_ids,
            struct_subset=[struct],
            derivative=derivative,
            as_mesh=True,
        )
        + ppdict.ExtractUniqueKey(nested=True)
        + ppdict.DictMap(ppdict.Subsample(subsample))
    )

    protocol = PairwiseVarifold(mesh_loader, known_correspondences, results_dir)

    protocol.run()


if __name__ == "__main__":
    logging.basicConfig(
        filename="run.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

    structs = get_all_first_structs(order=True, include_brstem=False)
    derivative = "enigma"

    data_dir = (
        "/scratch/data/maternal" if putils.in_frank() else "~/.herbrain/data/maternal"
    )

    for struct in structs:
        results_dir = Path("results") / f"{struct}_{derivative}"
        if results_dir.exists():
            logging.info(f"Skipping {struct} because folder already exists")
            continue

        results_dir.mkdir(parents=True, exist_ok=False)

        try:
            protocol_per_struct(
                struct,
                subject_ids=get_subject_ids(sort=True),
                data_dir=data_dir,
                results_dir=results_dir,
                subsample=None,  # make None to run all
            )
        except Exception as e:
            logging.warning(f"Oops, something went wrong for {struct}: {e}")

import json
from pathlib import Path

import polpo.preprocessing.dict as ppdict
from polpo.preprocessing.load.pregnancy.jacobs import (
    MeshLoader,
    get_subject_ids,
)
from polpo.protocol.varifold import PairwiseVarifold

if __name__ == "__main__":
    STRUCT = "L_Hipp"
    SUBJECT_IDS = get_subject_ids(sort=True)
    DERIVATIVE = "enigma"
    DATA_DIR = "~/.herbrain/data/maternal"

    RESULTS_DIR = Path("results") / f"{STRUCT}_{DERIVATIVE}"

    SUBSAMPLE = 1  # make None to run all

    RESULTS_DIR.mkdir(parents=True, exist_ok=False)

    # for serialization
    params = dict(
        struct=STRUCT,
        subject_ids=SUBJECT_IDS,
        derivative=DERIVATIVE,
        data_dir=DATA_DIR,
    )
    with open(RESULTS_DIR / "params.json", "w") as file:
        json.dump(params, file, indent=4)

    known_correspondences = True if DERIVATIVE == "enigma" else False

    mesh_loader = (
        MeshLoader(
            data_dir=DATA_DIR,
            subject_subset=SUBJECT_IDS,
            struct_subset=[STRUCT],
            derivative=DERIVATIVE,
            as_mesh=True,
        )
        + ppdict.ExtractUniqueKey(nested=True)
        + ppdict.DictMap(ppdict.Subsample(SUBSAMPLE))
    )

    protocol = PairwiseVarifold(mesh_loader, known_correspondences, RESULTS_DIR)

    protocol.run()

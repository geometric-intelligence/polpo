import json
import logging
from pathlib import Path

import polpo.preprocessing.dict as ppdict
import polpo.utils as putils
from polpo.enigma.naming import get_all_structs
from polpo.jacobs.mesh import MeshDatasetLoader
from polpo.jacobs.utils import get_subject_ids
from polpo.protocol.pairwise_varifold import PairwiseVarifold


def protocol_per_struct(
    struct,
    subject_ids,
    data_dir,
    results_dir,
    derivative="enigma",
    ratio_charlen_mesh=2.0,
    ratio_charlen=0.25,
    n_jobs=1,
    backend="keops",
):
    # for serialization
    params = dict(
        struct=struct,
        subject_ids=subject_ids,
        derivative=derivative,
        data_dir=data_dir,
    )
    with open(results_dir / "extra_params.json", "w") as file:
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

    protocol = PairwiseVarifold(
        known_correspondences,
        results_dir,
        ratio_charlen_mesh=ratio_charlen_mesh,
        ratio_charlen=ratio_charlen,
        n_jobs=n_jobs,
        backend=backend,
    )

    protocol.run(dataset)


if __name__ == "__main__":
    logging.basicConfig(
        filename="run.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

    structs = get_all_structs(order=True, include_brstem=False)
    derivative = "enigma"

    data_dir = (
        "/scratch/data/maternal"
        if putils.in_frank()
        else Path.home() / ".herbrain/data/maternal"
    )

    for struct in structs:
        results_dir = (
            putils.get_results_path()
            / "pairwise_varifold"
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
                subject_ids=get_subject_ids(sort=True),
                data_dir=data_dir,
                results_dir=results_dir,
                ratio_charlen_mesh=2.0,
                ratio_charlen=0.25,
                n_jobs=1,
                backend="keops",
            )
        except Exception as e:
            logging.warning(f"Oops, something went wrong for {struct}: {e}")

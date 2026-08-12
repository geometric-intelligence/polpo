import polpo.utils as putils
from polpo.workflow.lddmm_to_global.distances import (
    EuclideanDistances,
    PersistentEvaluator,
    VarifoldDistances,
)

if __name__ == "__main__":
    outputs_dir = putils.get_results_path() / "blobs/lddmm_to_global"

    PersistentEvaluator(
        VarifoldDistances(outputs_dir),
        "post_dists_var",
    ).run()

    PersistentEvaluator(
        EuclideanDistances(outputs_dir),
        "post_dists_euc",
    ).run()

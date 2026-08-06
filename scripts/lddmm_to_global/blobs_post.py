import polpo.utils as putils
from polpo.workflow.lddmm_to_global.distances import VarifoldDistances

if __name__ == "__main__":
    outputs_dir = putils.get_results_path() / "blobs/lddmm_to_global"

    VarifoldDistances(outputs_dir).run(overwrite=False)

import polpo.utils as putils
from polpo.workflow.lddmm_to_global.post import LddmmToGlobalDistances

if __name__ == "__main__":
    outputs_dir = putils.get_results_path() / "blobs/lddmm_to_global"

    LddmmToGlobalDistances(outputs_dir).run(overwrite=False)

import polpo.utils as putils
from polpo.lddmm_to_global.post import PostDistances

if __name__ == "__main__":
    outputs_dir = putils.get_results_path() / "blobs/lddmm_to_global"

    PostDistances(outputs_dir).run(overwrite=False)

import polpo.utils as putils
from polpo.lddmm_to_global.post import post_dists

if __name__ == "__main__":
    outputs_dir = putils.get_results_path() / "blobs/lddmm_to_global"

    post_dists(outputs_dir)

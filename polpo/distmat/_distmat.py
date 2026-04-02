import numpy as np


def sort_dist_mat(mat):
    dist_norms = np.linalg.norm(mat, axis=-1)
    sorted_idx = np.argsort(-dist_norms, axis=-1)

    perm_dists = mat[sorted_idx][:, sorted_idx]

    return perm_dists, sorted_idx

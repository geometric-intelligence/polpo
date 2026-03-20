import numpy as np

from polpo.auto_all import auto_all

# TODO: rename to array later?


def pairwise_dists(points, dist_fnc):
    dists = []
    for index, point in enumerate(points):
        # TODO: do vectorize version of it?
        for cmp_point in points[index + 1 :]:
            dists.append(dist_fnc(point, cmp_point))

    return triu_vec_to_sym(np.array(dists))


def sym_to_triu_vec(mat, k=1):
    return mat[np.triu_indices(len(mat), k=k)]


def triu_vec_to_sym(vec, includes_diag=False):
    if includes_diag:
        k = 0
        n = int((np.sqrt(8 * vec.size + 1) - 1) / 2)
    else:
        k = 1
        n = int((1 + np.sqrt(1 + 8 * vec.size)) / 2)

    mat = np.zeros((n, n))

    mat[np.triu_indices(n, k=k)] = vec

    mat = mat + mat.T

    if includes_diag == 0:
        mat = mat - diag(mat)

    return mat


def diag(mat):
    return mat[np.diag_indices(mat.shape[-1])]


def get_diag_blocks_by_size(mat, sizes):
    indices = np.r_[[0], np.cumsum(sizes)]

    return [
        mat[init_index:end_index, init_index:end_index]
        for init_index, end_index in zip(indices, indices[1:])
    ]


__all__ = auto_all(globals())

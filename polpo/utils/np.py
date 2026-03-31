import numpy as np

try:
    # avoid making dependency joblib explicit
    from joblib import Parallel, delayed
except ImportError:
    pass

from polpo.auto_all import auto_all


def pairwise_dists(points, dist_fnc, as_matrix=True):
    # TODO: add notion of backend?
    dists = []
    for index, point in enumerate(points):
        # TODO: do vectorize version of it?
        for cmp_point in points[index + 1 :]:
            dists.append(dist_fnc(point, cmp_point))

    dists = np.array(dists)
    if as_matrix:
        return triu_vec_to_sym(dists)

    return dists


def pairwise_dists_par(
    points, dist_fnc, as_matrix=True, n_jobs=None, verbose=0, prefer="threads"
):
    if n_jobs == 1:
        return pairwise_dists(points, dist_fnc, as_matrix=as_matrix)

    row_ind, col_ind = np.triu_indices(len(points), k=1)

    dists = Parallel(n_jobs=n_jobs, verbose=verbose, prefer=prefer)(
        delayed(dist_fnc)(points[i], points[j]) for i, j in zip(row_ind, col_ind)
    )

    dists = np.array(dists)
    if as_matrix:
        return triu_vec_to_sym(dists)

    return dists


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

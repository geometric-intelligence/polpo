import numpy as np

from polpo.pymetis import partition_graph

from .topology import compute_vertex_adjacency


def partition_vertices_balanced(faces, n_parts):
    """Partition mesh vertices into approximately equal-sized connected patches.

    The mesh is represented by its vertex adjacency graph and partitioned using
    METIS. The partitioning approximately balances the number of vertices
    across patches while minimizing the number of mesh edges crossing between
    patches.

    Parameters
    ----------
    faces : array-like, shape (n_faces, 3)
        Vertex indices of triangular faces.
    n_parts : int
        Number of vertex partitions.

    Returns
    -------
    labels : ndarray, shape (n_vertices,)
        Partition label assigned to each vertex. Labels are integers in
        ``[0, n_parts)``.
    """
    adj_mat = compute_vertex_adjacency(faces)

    _, labels = partition_graph(adj_mat, n_parts)

    return np.asarray(labels)


def labels_to_vertex_partitions(labels):
    """Convert partition labels to index arrays.

    Parameters
    ----------
    labels : array-like
        Partition label for each item.

    Returns
    -------
    vertex_partitions: list of ndarray
        Indices belonging to each partition.
    """
    return [np.flatnonzero(labels == label) for label in np.unique(labels)]

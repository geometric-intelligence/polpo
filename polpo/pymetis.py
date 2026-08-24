import pymetis


def partition_graph(adj_mat, n_parts):
    """Partition a graph into approximately balanced connected parts.

    Parameters
    ----------
    adj_mat : scipy.sparse.csr_matrix, shape (n_vertices, n_vertices)
        Sparse adjacency matrix of the graph.
    n_parts : int
        Number of partitions.

    Returns
    -------
    n_edge_cuts : int
        Number of graph edges crossing between different partitions.
    labels : array-like, shape (n_vertices,)
        Partition label assigned to each vertex.
    """
    adjacency = pymetis.CSRAdjacency(
        adj_mat.indptr,
        adj_mat.indices,
    )

    n_edge_cuts, labels = pymetis.part_graph(
        n_parts,
        adjacency=adjacency,
        options=pymetis.Options(contig=True),
    )
    return n_edge_cuts, labels

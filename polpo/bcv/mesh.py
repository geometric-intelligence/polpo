import numpy as np

from polpo.surface_mesh.partition import labels_to_vertex_partitions


def compute_held_out_cols(labels, dim=3):
    """Compute held-out row indices from outer-group identifiers.

    Each fold holds out all rows belonging to a combination of
    ``n_outer`` distinct outer groups.

    Parameters
    ----------
    outer_ids : array-like
        Outer-group identifier associated with each row.
    n_outer : int
        Number of outer groups to hold out in each fold.

    Returns
    -------
    held_out_cols : dict
        Mapping from tuples of held-out outer-group identifiers to the
        corresponding row indices.
    """
    return [
        vertices_to_cols(vertices, dim=dim)
        for vertices in labels_to_vertex_partitions(labels)
    ]


def vertices_to_cols(vertices, dim=3):
    """Convert vertex indices to flattened feature-column indices.

    Assumes features are obtained by flattening an array of shape
    ``(n_vertices, dim)`` in C order.

    Parameters
    ----------
    vertices : array-like
        Vertex indices.
    dim : int
        Number of features per vertex.

    Returns
    -------
    cols : ndarray
        Corresponding indices in the flattened feature array.
    """
    return (dim * vertices[:, None] + np.arange(dim)).ravel()

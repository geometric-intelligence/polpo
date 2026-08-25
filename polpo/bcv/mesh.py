import numpy as np

from polpo.bcv import BCVBlock
from polpo.bcv.folds import compute_held_out_blocks
from polpo.bcv.grouped import compute_held_out_rows, group_ids_from_sizes
from polpo.bcv.model_selection import (
    select_rank_min_error,
    select_rank_one_se,
    select_rank_one_se_grouped,
)
from polpo.dataset import Dataset
from polpo.seed import resolve_seed
from polpo.surface_mesh.partition import (
    labels_to_vertex_partitions,
    partition_vertices_balanced,
)


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
    seed : int or None
        Random seed used for mesh partitioning. If None, a seed is generated at
        runtime.

    Returns
    -------
    cols : ndarray
        Corresponding indices in the flattened feature array.
    """
    return (dim * vertices[:, None] + np.arange(dim)).ravel()


class GroupedMeshRankSelection:
    def __init__(self, n_parts=10, n_groups=1, center=False, seed=None):
        self.n_parts = n_parts
        self.n_groups = n_groups
        self.center = center

        self.seed = seed

    def _compute_held_out_cols(self, mesh_faces):
        seed = self.seed_ = resolve_seed(self.seed)
        partition_labels = partition_vertices_balanced(
            mesh_faces, n_parts=self.n_parts, seed=seed
        )

        return {
            index: cols
            for index, cols in enumerate(compute_held_out_cols(partition_labels))
        }

    def _compute_held_out_rows(self, dataset):
        sizes = dataset.reduce_outer(lambda x: len(x))
        group_ids = group_ids_from_sizes(sizes.values_list(), sizes.keys_list())
        return compute_held_out_rows(group_ids, n_groups=self.n_groups)

    def fit(self, mesh_faces, dataset):
        held_out_cols = self._compute_held_out_cols(mesh_faces)
        held_out_rows = self._compute_held_out_rows(dataset)

        held_out = compute_held_out_blocks(held_out_rows, held_out_cols)

        X = np.concatenate(
            dataset.reduce_outer(lambda values: np.stack(values)).values_list()
        )

        blocks = Dataset(
            {
                key: BCVBlock(center=self.center).fit(X, rows, cols)
                for key, (rows, cols) in held_out.items()
            }
        )

        n_rank = blocks.map_values(lambda block: block.s_.shape[0]).values_list()

        self.errors_ = blocks.map_values(
            lambda block: block.errors(
                max_rank=min(n_rank),
                normalize=True,
            )
        )

        fold_errors = self.errors_array_
        fold_errors_grouped = self.errors_array_grouped_

        self.rank_min_error_ = select_rank_min_error(fold_errors)
        self.rank_one_se_ = select_rank_one_se(fold_errors)
        self.rank_one_se_grouped_ = select_rank_one_se_grouped(fold_errors_grouped)
        self.rank_ = self.rank_one_se_grouped_

        self.held_out_cols_ = held_out_cols
        self.held_out_rows_ = held_out_rows

        self.blocks_ = blocks

        return self

    @property
    def errors_array_(self):
        return self.errors_.apply(lambda values: np.stack(values))

    @property
    def errors_array_grouped_(self):
        return (
            self.errors_.nest()
            .reduce_outer(lambda values: np.stack(values))
            .apply(lambda values: np.stack(values))
        )

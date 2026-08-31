from pathlib import Path

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
from polpo.io.json import load_json, save_json
from polpo.seed import resolve_seed
from polpo.surface_mesh.partition import (
    labels_to_vertex_partitions,
    partition_vertices_balanced,
)
from polpo.workflow.task import TaskRunner, task


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

        self.held_out_cols_ = held_out_cols
        self.held_out_rows_ = held_out_rows

        self.blocks_ = blocks

        return self

    @property
    def rank_(self):
        return self.rank_one_se_grouped_

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


class GroupedMeshRankSelectionResult:
    """Results of mesh bi-cross-validation rank selection.

    Parameters
    ----------
    errors : ndarray, shape (n_folds, n_ranks)
        Cross-validation errors for each held-out block and candidate rank.
    keys : list
        Keys identifying the folds represented by the rows of ``errors``.
    seed : int
        Random seed used to construct the mesh partitions.
    """

    def __init__(
        self,
        errors,
        keys,
        n_parts,
        n_groups,
        center,
        seed,
    ):
        self.errors = errors
        self.keys = keys

        self.n_parts = n_parts
        self.n_groups = n_groups
        self.center = center
        self.seed = seed

    @property
    def rank(self):
        return self.rank_one_se_grouped

    @property
    def rank_min_error(self):
        return select_rank_min_error(self.errors)

    @property
    def rank_one_se(self):
        return select_rank_one_se(self.errors)

    @property
    def rank_one_se_grouped(self):
        return select_rank_one_se_grouped(self.errors_grouped)

    @property
    def errors_grouped(self):
        return self.errors.reshape(-1, self.n_parts, self.errors.shape[-1])

    @classmethod
    def from_selection(cls, selection):
        """Create results from a fitted ``GroupedMeshRankSelection``."""
        return cls(
            errors=selection.errors_array_,
            keys=selection.errors_.keys_list(),
            seed=selection.seed_,
            center=selection.center,
            n_parts=selection.n_parts,
            n_groups=selection.n_groups,
        )

    def to_dir(self, results_dir):
        """Write results to disk."""
        results_dir.mkdir(parents=True, exist_ok=True)

        np.save(results_dir / "errors.npy", self.errors)

        save_json(
            results_dir / "params.json",
            {
                "n_parts": self.n_parts,
                "n_groups": self.n_groups,
                "center": self.center,
                "seed": self.seed,
                "keys": self.keys,
            },
        )

        return self

    @classmethod
    def from_dir(cls, results_dir):
        """Load rank-selection results from disk."""
        errors = np.load(results_dir / "errors.npy")
        params = load_json(results_dir / "params.json")

        return cls(
            errors=errors,
            **params,
        )


class _BaseGroupedMeshRankSelectionRunner(TaskRunner):
    def __init__(
        self,
        results_dir,
        state_dir=None,
        **selection_kwargs,
    ):
        if state_dir is None:
            state_dir = (
                Path(".rank_selection") if results_dir is None else Path(results_dir)
            )

        super().__init__(state_dir)

        self.results_dir = results_dir
        self.selection_kwargs = selection_kwargs

    @task
    def select_rank(self):
        mesh_faces, dataset = self.prepare_inputs()

        selection = GroupedMeshRankSelection(
            **self.selection_kwargs,
        ).fit(mesh_faces, dataset)

        results = GroupedMeshRankSelectionResult.from_selection(selection)
        results.to_dir(self.results_dir)


class LazyGroupedMeshRankSelectionRunner(_BaseGroupedMeshRankSelectionRunner):
    def __init__(
        self,
        prepare_inputs,
        results_dir,
        state_dir=None,
        **selection_kwargs,
    ):
        super().__init__(results_dir, state_dir, **selection_kwargs)
        self._prepare_inputs = prepare_inputs

    def prepare_inputs(self):
        return self._prepare_inputs()


class GroupedMeshRankSelectionRunner(_BaseGroupedMeshRankSelectionRunner):
    def __init__(
        self,
        mesh_faces,
        dataset,
        results_dir,
        state_dir=None,
        **selection_kwargs,
    ):
        super().__init__(results_dir, state_dir, **selection_kwargs)
        self.mesh_faces = mesh_faces
        self.dataset = dataset

    def prepare_inputs(self):
        return self.mesh_faces, self.dataset

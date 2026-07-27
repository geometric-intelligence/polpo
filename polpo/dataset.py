from collections.abc import Mapping

import numpy as np

from polpo.utils.dict_ import nest_dict, unnest_dict


class DatasetMapping(Mapping):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"{type(self).__name__}({self.data!r})"


class Dataset(DatasetMapping):
    def __init__(self, data):
        self.data = data

        self._sep = "-"

    def _new(self, data):
        return type(self)(data)

    def as_dict(self):
        return self.data

    def keys_list(self):
        return list(self.data.keys())

    def values_list(self):
        return list(self.data.values())

    def with_values(self, values):
        # uses same keys
        data = dict(zip(self.data.keys(), values))
        return Dataset(data)

    def nest(self):
        data = nest_dict(self.data, sep=self._sep)
        return NestedDataset(data)

    def map_values(self, func, /, *args, **kwargs):
        """Apply ``func`` independently to every dataset value.

        Parameters
        ----------
        func : callable
            Function applied to each value.
        *args
            Additional positional arguments passed to ``func``.
        **kwargs
            Additional keyword arguments passed to ``func``.

        Returns
        -------
        Dataset
            Dataset with the same keys and transformed values.
        """
        data = {key: func(value, *args, **kwargs) for key, value in self.data.items()}
        return self._new(data)

    def apply(self, func, /, *args, **kwargs):
        """Apply ``func`` once to the ordered dataset values."""
        return func(self.values_list(), *args, **kwargs)

    def transform(self, func, /, *args, **kwargs):
        """Apply ``func`` to all values and preserve the dataset keys."""
        return self.with_values(self.apply(func, *args, **kwargs))

    def sample(self, n_samples=1, *, random_state=None):
        """Sample dataset entries without replacement.

        Parameters
        ----------
        n_samples : int
            Number of entries to sample.
        random_state : int or numpy.random.Generator, optional
            Random seed or generator.

        Returns
        -------
        Dataset
            Dataset containing the sampled entries.
        """
        if n_samples < 1:
            raise ValueError("n_samples must be positive.")

        if n_samples > len(self):
            raise ValueError(
                f"Cannot sample {n_samples} entries from a dataset "
                f"containing {len(self)}."
            )

        rng = np.random.default_rng(random_state)

        sampled_keys = rng.choice(
            self.keys_list(),
            size=n_samples,
            replace=False,
        )

        return self._new({key: self[key] for key in sampled_keys})


class NestedDataset(DatasetMapping):
    def __init__(self, data):
        self.data = data

        self._sep = "-"

    def _new(self, data):
        return type(self)(data)

    def as_dict(self):
        return self.data

    def flatten(self):
        data = unnest_dict(self.data, sep=self._sep)
        return Dataset(data)

    def transform(self, func, /, *args, **kwargs):
        """Apply a function to the flattened dataset and restore its structure.

        The dataset is flattened into an ordered list of values and passed as the
        first argument to ``func``.
        The values returned by ``func`` are associated
        with the original keys and converted back into a nested dataset.

        Parameters
        ----------
        func : callable
            Function applied to the flattened values. Its first argument must
            accept the list of dataset values.
        *args
            Additional positional arguments forwarded to ``func``.
        **kwargs
            Additional keyword arguments forwarded to ``func``.

        Returns
        -------
        NestedDataset
            A nested dataset containing the values returned by ``func``.
        """
        return self.flatten().transform(func, *args, **kwargs).nest()

    def map_values(self, func, /, *args, **kwargs):
        """Apply ``func`` independently to every inner value.

        Parameters
        ----------
        func : callable
            Function applied to each inner value.
        *args
            Additional positional arguments passed to ``func``.
        **kwargs
            Additional keyword arguments passed to ``func``.

        Returns
        -------
        NestedDataset
            Dataset with the same keys and transformed values.
        """
        data = {
            outer_key: {
                inner_key: func(value, *args, **kwargs)
                for inner_key, value in inner_data.items()
            }
            for outer_key, inner_data in self.data.items()
        }
        return self._new(data)

    def map_items(self, func, /, *args, **kwargs):
        """Apply ``func`` to each outer key, inner key, and value."""
        data = {
            outer_key: {
                inner_key: func(
                    outer_key,
                    inner_key,
                    value,
                    *args,
                    **kwargs,
                )
                for inner_key, value in inner_data.items()
            }
            for outer_key, inner_data in self.items()
        }
        return self._new(data)

    def reduce_outer(self, func, /, *args, **kwargs):
        """Apply ``func`` to each outer dataset and return one result per key."""
        return Dataset(
            {
                outer_key: func(list(inner_data.values()), *args, **kwargs)
                for outer_key, inner_data in self.data.items()
            }
        )

    def sample_inner(self, n_samples=1, *, random_state=None):
        """Sample inner entries independently for each outer key.

        Parameters
        ----------
        n_samples : int
            Number of inner entries sampled per outer key.
        random_state : int or numpy.random.Generator
            Random seed or generator.

        Returns
        -------
        NestedDataset
            Dataset containing the sampled inner entries.
        """
        if n_samples < 1:
            raise ValueError("n_samples must be positive.")

        rng = np.random.default_rng(random_state)

        data = {}

        for outer_key, inner_data in self.items():
            inner_keys = list(inner_data)

            if n_samples > len(inner_keys):
                raise ValueError(
                    f"Cannot sample {n_samples} entries from "
                    f"{outer_key!r}, which contains {len(inner_keys)}."
                )

            sampled_keys = rng.choice(
                inner_keys,
                size=n_samples,
                replace=False,
            )

            data[outer_key] = {
                inner_key: inner_data[inner_key] for inner_key in sampled_keys
            }

        return self._new(data)

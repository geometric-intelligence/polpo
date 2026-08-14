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
        data = nest_dict(self.data)
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

    def map_keys(self, func, /, *args, on_collision="raise", **kwargs):
        """Apply ``func`` to each key while preserving values."""
        data = {}

        for key, value in self.items():
            new_key = func(key, *args, **kwargs)

            if new_key in data:
                if on_collision == "raise":
                    raise ValueError(
                        f"Key transformation produced duplicate key {new_key!r}."
                    )
                if on_collision == "keep_first":
                    continue
                if on_collision != "keep_last":
                    raise ValueError(
                        "on_collision must be 'raise', 'keep_first', or 'keep_last'."
                    )

            data[new_key] = value

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

        keys = self.keys_list()
        indices = rng.choice(
            len(keys),
            size=n_samples,
            replace=False,
        )
        sampled_keys = [keys[index] for index in indices]

        return self._new({key: self[key] for key in sampled_keys})

    def select(self, keys, *, ignore_missing=False):
        """Return a dataset restricted to the given keys.

        Parameters
        ----------
        keys : iterable
            Keys to select. Their order determines the order of the returned
            dataset.

        Returns
        -------
        Dataset
            Dataset containing the selected entries.
        """
        if ignore_missing:
            return self._new({key: self[key] for key in keys if key in self})

        return self._new({key: self[key] for key in keys})

    def filter_values(self, predicate):
        return self._new(
            dict((key, value) for key, value in self.items() if predicate(value))
        )

    @classmethod
    def merge(cls, datasets):
        data = {}

        for dataset in datasets:
            overlap = data.keys() & dataset.keys()
            if overlap:
                raise ValueError(f"Duplicate keys: {overlap}")

            data.update(dataset.items())

        return cls(data)


class NestedDataset(DatasetMapping):
    def __init__(self, data):
        self.data = data

    def _new(self, data):
        return type(self)(data)

    def keys_list(self):
        return list(self.data.keys())

    def inner_keys(self):
        return {outer_key: list(inner) for outer_key, inner in self.items()}

    def nested_keys(self):
        return [
            (outer_key, inner_key)
            for outer_key, inner in self.items()
            for inner_key in inner
        ]

    def as_dict(self):
        return self.data

    def flatten(self):
        data = unnest_dict(self.data, sep=None)
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

    def map_keys(self, func, /, *args, **kwargs):
        data = {}

        for outer_key, inner_data in self.items():
            for inner_key, value in inner_data.items():
                new_outer_key, new_inner_key = func(
                    outer_key,
                    inner_key,
                    *args,
                    **kwargs,
                )
                data.setdefault(new_outer_key, {})[new_inner_key] = value

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

            indices = rng.choice(
                len(inner_keys),
                size=n_samples,
                replace=False,
            )
            sampled_keys = [inner_keys[index] for index in indices]

            data[outer_key] = {
                inner_key: inner_data[inner_key] for inner_key in sampled_keys
            }

        return self._new(data)

    def filter_keys(self, predicate, /, *args, **kwargs):
        """Filter entries according to their outer and inner keys.

        Parameters
        ----------
        predicate : callable
            Function called as ``predicate(outer_key, inner_key, *args, **kwargs)``.
            Entries for which it returns ``True`` are retained.
        *args
            Additional positional arguments passed to ``predicate``.
        **kwargs
            Additional keyword arguments passed to ``predicate``.

        Returns
        -------
        NestedDataset
            Dataset containing the selected entries.
        """
        data = {
            outer_key: {
                inner_key: value
                for inner_key, value in inner_data.items()
                if predicate(outer_key, inner_key, *args, **kwargs)
            }
            for outer_key, inner_data in self.items()
        }

        return self._new(
            {
                outer_key: inner_data
                for outer_key, inner_data in data.items()
                if inner_data
            }
        )

    def drop_outer(self, keys):
        keys = set(keys)
        return self._new(
            {
                outer_key: inner
                for outer_key, inner in self.items()
                if outer_key not in keys
            }
        )

    def split_outer(self):
        """Split into one Dataset per outer key."""
        return {outer_key: self.get_outer(outer_key) for outer_key in self}

    def iter_outer(self):
        for outer_key in self:
            yield outer_key, self.get_outer(outer_key)

    def get_outer(self, outer_key):
        """Return the inner dataset associated with an outer key."""
        return Dataset(self.data[outer_key])

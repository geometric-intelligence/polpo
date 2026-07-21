from collections.abc import Mapping

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

    def apply_flat(self, func, /, *args, **kwargs):
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
        flat = self.flatten()
        output = func(flat.values_list(), *args, **kwargs)
        return flat.with_values(output).nest()

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

    def reduce_outer(self, func, /, *args, **kwargs):
        """Apply ``func`` to each outer dataset and return one result per key."""
        return Dataset(
            {
                outer_key: func(list(inner_data.values()), *args, **kwargs)
                for outer_key, inner_data in self.data.items()
            }
        )

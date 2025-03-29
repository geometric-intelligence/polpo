from tqdm import tqdm

from polpo.collections import swap_nested_dict

from ._preprocessing import (
    Filter,
    IdentityStep,
    StepWrappingPreprocessingStep,
    _wrap_step,
)
from .base import PreprocessingStep


class Hash(PreprocessingStep):
    def __init__(self, key_index=0, ignore_none=False, ignore_empty=False):
        super().__init__()
        self.key_index = key_index
        self.ignore_none = ignore_none
        self.ignore_empty = ignore_empty

    def apply(self, data):
        new_data = {}
        for datum in data:
            if not isinstance(datum, list):
                datum = list(datum)

            key = datum.pop(self.key_index)

            if len(datum) == 1:
                datum = datum[0]

            if (self.ignore_none and datum is None) or (
                self.ignore_empty and isinstance(datum, list) and len(datum) == 0
            ):
                continue

            new_data[key] = datum

        return new_data


class DictMerger(PreprocessingStep):
    # NB: not shared keys are ignored

    def _collect_shared_keys(self, data):
        keys = set(data[0].keys())
        for datum in data[1:]:
            keys = keys.intersection(set(datum.keys()))

        return keys

    def apply(self, data):
        shared_keys = self._collect_shared_keys(data)
        out = []
        for key in shared_keys:
            out.append([datum[key] for datum in data])

        return out


class HashWithIncoming(StepWrappingPreprocessingStep):
    def __init__(self, step=None, key_step=None):
        super().__init__(step)
        self.key_step = _wrap_step(key_step)

    def apply(self, keys_data):
        values_data = self.step(keys_data)
        keys_data = self.key_step(keys_data)

        return {key: value for key, value in zip(keys_data, values_data)}


class DictFilter(Filter):
    def __init__(self, func, filter_keys=False):
        if filter_keys:
            func_ = lambda x: func(x[0])
        else:
            func_ = lambda x: func(x[1])
        super().__init__(func_, collection_type=dict, to_iter=lambda x: x.items())


class SelectKeySubset(DictFilter):
    """Select key subset.

    Parameters
    ----------
    subset : array-like
        Subset of keys to consider.
    keep : bool
        Whether to keep or exclude subset.
    """

    def __init__(self, subset, keep=True):
        if keep:
            func = lambda key: key in subset
        else:
            func = lambda key: key not in subset

        super().__init__(func, filter_keys=True)

    def __new__(cls, subset, keep=True):
        """Instantiate step.

        If subset is None, then it instantiates the identity step.
        Syntax sugar to simplify control flow.
        """
        if subset is None:
            return IdentityStep()

        return super().__new__(cls)


class DictNoneRemover(Filter):
    def __init__(self):
        super().__init__(
            func=lambda x: x[1] is not None,
            collection_type=dict,
            to_iter=lambda x: x.items(),
        )


class DictExtractKey(PreprocessingStep):
    def __init__(self, key):
        super().__init__()
        self.key = key

    def apply(self, data):
        return data[self.key]


class DictToValuesList(PreprocessingStep):
    def apply(self, data):
        return list(data.values())


class DictToTuplesList(PreprocessingStep):
    def apply(self, data):
        return list(zip(data.keys(), data.values()))


class DictMap(StepWrappingPreprocessingStep):
    """Apply a given step to each element of a dictionary.

    Parameters
    ----------
    step : callable
        Preprocessing step to apply to value.
    pbar : bool
        Whether to show a progress bar.
    key_step : callable
        Preprocessing step to apply to key.
    special_keys : array-like
        Keys to be subject to a different step.
    special_step : callable
        Step to apply to special keys.
    """

    def __init__(
        self, step=None, pbar=False, key_step=None, special_keys=(), special_step=None
    ):
        super().__init__(step)
        self.pbar = pbar
        self.key_step = _wrap_step(key_step)
        self.special_keys = special_keys
        self.special_step = _wrap_step(special_step)

    def apply(self, data):
        """Apply step.

        Parameters
        ----------
        data : dict

        Returns
        -------
        new_data : dict
        """
        out = {}
        for key, value in tqdm(data.items(), disable=not self.pbar):
            if key in self.special_keys:
                value_ = self.special_step(value)
            else:
                value_ = self.step(value)
            out[self.key_step(key)] = value_

        return out


class NestedDictSwapper(PreprocessingStep):
    def apply(self, nested_dict):
        return swap_nested_dict(nested_dict)

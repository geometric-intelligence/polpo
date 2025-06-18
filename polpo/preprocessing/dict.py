import warnings

from joblib import Parallel, delayed
from tqdm import tqdm

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

    def __call__(self, data):
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

    def __call__(self, data):
        shared_keys = self._collect_shared_keys(data)
        out = []
        for key in shared_keys:
            out.append([datum[key] for datum in data])

        return out


class HashWithIncoming(StepWrappingPreprocessingStep):
    def __init__(self, step=None, key_step=None):
        super().__init__(step)
        self.key_step = _wrap_step(key_step)

    def __call__(self, keys_data):
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

    def __call__(self, data):
        return data[self.key]


class DictToValuesList(PreprocessingStep):
    def __call__(self, data):
        return list(data.values())


class ValuesListToDict(PreprocessingStep):
    def __init__(self, keys=None):
        super().__init__()
        self.keys = keys

    def __call__(self, data):
        if isinstance(data, tuple) and len(data) == 2:
            keys, values = data
        else:
            keys = self.keys
            values = data

        return dict(zip(keys, values))


class DictToTuplesList(PreprocessingStep):
    def __call__(self, data):
        return list(zip(data.keys(), data.values()))


class DictUpdate(PreprocessingStep):
    def __call__(self, data):
        new_data = data[0].copy()
        for datum in data[1:]:
            new_data.update(datum)

        return new_data


class SerialDictMap(StepWrappingPreprocessingStep):
    """Apply a given step to each element of a dict.

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

    def __call__(self, data):
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


class ParDictMap(StepWrappingPreprocessingStep):
    """Apply a given step to each element of a dict in parallel.

    Parameters
    ----------
    step : callable
        Preprocessing step to apply to value.
    n_jobs : int
        The maximum number of concurrently running jobs.
    verbose : int
        The verbosity level: if non zero, progress messages are
        printed. Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.

    Notes
    -----
    * has less functionality than `DictMap`
    """

    def __init__(self, step, n_jobs=-1, verbose=0):
        super().__init__(step)
        self.n_jobs = n_jobs
        self.verbose = verbose

    def __call__(self, data):
        with Parallel(n_jobs=self.n_jobs, verbose=self.verbose) as parallel:
            res = parallel(delayed(self.step)(datum) for datum in data.items())

        return dict(zip(data.keys(), res))


class DictMap:
    """Apply a given step to each element of a dict."""

    def __new__(
        cls,
        step=None,
        pbar=False,
        key_step=None,
        special_keys=(),
        special_step=None,
        n_jobs=0,
        verbose=0,
    ):
        """Instantiate class.

        Parameters
        ----------
        step : callable
            Preprocessing step to apply to value.
        pbar : bool
            Whether to show a progress bar. Only if serial.
        key_step : callable
            Preprocessing step to apply to key. Only if serial.
        special_keys : array-like
            Keys to be subject to a different step.  Only if serial.
        special_step : callable
            Step to apply to special keys.  Only if serial.
        n_jobs : int
            The maximum number of concurrently running jobs.
        verbose : int
            The verbosity level. Only if parallel.
        """
        if n_jobs != 0:
            if pbar or key_step is not None or special_keys or special_step:
                warnings.warn(
                    "Several arguments where ignored, check `ParDictMap` for more info"
                )

            return ParDictMap(step, n_jobs=n_jobs, verbose=verbose)

        return SerialDictMap(
            step=step,
            pbar=pbar,
            key_step=key_step,
            special_keys=special_keys,
            special_step=special_step,
        )


class NestedDictSwapper(PreprocessingStep):
    def __call__(self, nested_dict):
        return {
            outer_key: {
                inner_key: nested_dict[inner_key][outer_key]
                for inner_key in nested_dict
            }
            for outer_key in next(iter(nested_dict.values()))
        }


class ListDictSwapper(PreprocessingStep):
    # swap a list of dict
    # assumes same keys

    def __call__(self, ls):
        if len(ls) == 0:
            return {}

        keys = ls[0].keys()
        out = {key: [] for key in keys}
        for elem in ls:
            for key in keys:
                out[key].append(elem[key])

        return out


class ZipWithKeys(StepWrappingPreprocessingStep):
    """Apply step to the values of a dict.

    Then, get a list back, zip with original keys, and return a dict.
    """

    def __init__(self, step, check_length=True):
        super().__init__(step)
        self.check_length = check_length

    def __call__(self, dict_):
        keys = list(dict_.keys())
        values = list(dict_.values())
        new_values = self.step(values)

        if self.check_length and len(new_values) != len(keys):
            raise ValueError(
                f"Length mismatch: {len(keys)} keys but {len(new_values)} outputs"
            )

        return dict(zip(keys, new_values))

from tqdm import tqdm

from ._preprocessing import (
    Filter,
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
    def __init__(self, step, pre_hash_step=None):
        super().__init__(step)
        self.pre_hash_step = _wrap_step(pre_hash_step)

    def apply(self, data):
        new_data = self.step.apply(data)
        data = self.pre_hash_step(data)

        return {datum: new_datum for datum, new_datum in zip(data, new_data)}


class DictKeysFilter(Filter):
    def __init__(self, values, keep=True):
        if keep:
            func = lambda x: x[0] in values
        else:
            func = lambda x: x[0] not in values

        super().__init__(func, collection_type=dict)


class DictNoneRemover(Filter):
    def __init__(self):
        super().__init__(
            func=lambda x: x[1] is not None,
            collection_type=dict,
            to_iter=lambda x: x.items(),
        )


class DictToValuesList(PreprocessingStep):
    def apply(self, data):
        return list(data.values())


class DictToTuplesList(PreprocessingStep):
    def apply(self, data):
        return list(zip(data.keys(), data.values()))


class DictUpdate(PreprocessingStep):
    def apply(self, data):
        new_data = data[0].copy()
        for datum in data[1:]:
            new_data.update(datum)

        return new_data


class DictMap(StepWrappingPreprocessingStep):
    def __init__(
        self, step=None, pbar=False, key_step=None, special_keys=(), special_step=None
    ):
        super().__init__(step)
        self.pbar = pbar
        self.key_step = _wrap_step(key_step)
        self.special_keys = special_keys
        self.special_step = _wrap_step(special_step)

    def apply(self, data):
        out = {}
        for key, datum in tqdm(data.items(), disable=not self.pbar):
            if key in self.special_keys:
                value = self.special_step(datum)
            else:
                value = self.step(datum)
            out[self.key_step(key)] = value

        return out

import abc
import warnings

from joblib import Parallel, delayed
from tqdm import tqdm

from .base import Pipeline, PreprocessingStep


class StepWrappingPreprocessingStep(PreprocessingStep, abc.ABC):
    def __init__(self, step):
        super().__init__()
        if isinstance(step, (list, str)):
            step = Pipeline(step)

        self.step = step


class BranchingPipeline(PreprocessingStep):
    def __init__(self, branches, merger=None):
        if merger is None:
            merger = NestingSwapper()

        super().__init__()
        self.branches = branches
        self.merger = merger

    def apply(self, data):
        out = []
        for pipeline in self.branches:
            out.append(pipeline.apply(data))

        return self.merger(out)


class IdentityStep(PreprocessingStep):
    def apply(self, data=None):
        return data


class NestingSwapper(PreprocessingStep):
    def apply(self, data):
        return list(zip(*data))


class HashMerger(PreprocessingStep):
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


class IndexSelector(StepWrappingPreprocessingStep):
    def __init__(self, index=0, repeat=False, step=None):
        if step is None:
            step = IdentityStep()

        super().__init__(step)

        self.index = index
        self.repeat = repeat

    def apply(self, data):
        selected = self.step(data[self.index])
        if self.repeat:
            return [selected] * len(data)

        return selected


class IgnoreIndex(PreprocessingStep):
    def __init__(self, index=0, inplace=False):
        super().__init__()
        self.index = index
        self.inplace = inplace

    def apply(self, data):
        if not self.inplace:
            data = data.copy()

        data.pop(self.index)
        return data


class ListSqueeze(PreprocessingStep):
    def __init__(self, raise_=True):
        self.raise_ = raise_

    def apply(self, data):
        if len(data) != 1:
            if self.raise_:
                raise ValueError("Unsqueezable!")
            else:
                return data

        return data[0]


class HashWithIncoming(StepWrappingPreprocessingStep):
    def apply(self, data):
        new_data = self.step.apply(data)

        return {datum: new_datum for datum, new_datum in zip(data, new_data)}


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
                self.ignore_empty
                and isinstance(datum, (list, tuple))
                and len(datum) == 0
            ):
                continue

            new_data[key] = datum

        return new_data


class TupleWith(StepWrappingPreprocessingStep):
    def __init__(self, step, incoming_first=True):
        super().__init__(step)
        self.incoming_first = incoming_first

    def apply(self, data):
        new_data = self.step.apply(data)
        ordering = (data, new_data) if self.incoming_first else (new_data, data)
        return [(datum, datum_) for datum, datum_ in zip(*ordering)]


class TupleWithIncoming(TupleWith):
    def __init__(self, step):
        super().__init__(step, incoming_first=True)


class Sorter(PreprocessingStep):
    def apply(self, data):
        return sorted(data)


class Filter(PreprocessingStep):
    def __init__(self, func, collection_type=list, to_iter=None):
        if to_iter is None:
            to_iter = lambda x: x

        super().__init__()
        self.func = func
        self.collection_type = collection_type
        self.to_iter = to_iter

    def apply(self, data):
        return self.collection_type(filter(self.func, self.to_iter(data)))


class EmptyRemover(Filter):
    def __init__(self):
        super().__init__(func=lambda x: len(x) > 0)


class NoneRemover(Filter):
    def __init__(self):
        super().__init__(func=lambda x: x is not None)


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


class NoneSkipper(StepWrappingPreprocessingStep):
    def apply(self, data):
        if data is None:
            return data

        return self.step(data)


class EmptySkipper(StepWrappingPreprocessingStep):
    def apply(self, data):
        if len(data) == 0:
            return data

        return self.step(data)


class ToList(PreprocessingStep):
    # TODO: better naming?
    def apply(self, data):
        return [data]


class DictToValues(PreprocessingStep):
    # TODO: better naming
    def apply(self, data):
        return list(data.values())


class DictUpdate(PreprocessingStep):
    def apply(self, data):
        new_data = data[0].copy()
        for datum in data[1:]:
            new_data.update(datum)

        return new_data


class SerialMap(StepWrappingPreprocessingStep):
    def __init__(self, step, pbar=False):
        super().__init__(step)
        self.pbar = pbar

    def apply(self, data):
        return [self.step(datum) for datum in tqdm(data, disable=not self.pbar)]


class ParallelMap(StepWrappingPreprocessingStep):
    def __init__(self, step, n_jobs=-1, verbose=0):
        super().__init__(step)
        self.n_jobs = n_jobs
        self.verbose = verbose

    def apply(self, data):
        with Parallel(n_jobs=self.n_jobs, verbose=self.verbose) as parallel:
            res = parallel(delayed(self.step)(datum) for datum in data)

        return list(res)


class DecorateToIterable(StepWrappingPreprocessingStep):
    def apply(self, data):
        decorated = False
        if not isinstance(data, (list, tuple)):
            decorated = True
            data = [data]

        new_data = self.step(data)

        if decorated:
            return new_data[0]

        return new_data


class Map:
    def __new__(cls, step, n_jobs=0, verbose=0, force_iter=False):
        if n_jobs != 0:
            map_ = ParallelMap(step, n_jobs=n_jobs, verbose=verbose)

        else:
            map_ = SerialMap(step, pbar=verbose > 0)

        if force_iter:
            return DecorateToIterable(map_)

        return map_


class HashMap(StepWrappingPreprocessingStep):
    def __init__(self, step, pbar=False):
        super().__init__(step)
        self.pbar = pbar

    def apply(self, data):
        out = {}
        for key, datum in tqdm(data.items(), disable=not self.pbar):
            out[key] = self.step(datum)

        return out


class IndexMap(StepWrappingPreprocessingStep):
    # TODO: name is confusing
    def __init__(self, step, index=0):
        super().__init__(step)
        self.index = index

    def apply(self, data):
        data[self.index] = self.step(data[self.index])

        return data


class Truncater(PreprocessingStep):
    # useful for debugging

    def __init__(self, value):
        super().__init__()
        self.value = value

    def apply(self, data):
        return data[: self.value]


class DataPrinter(PreprocessingStep):
    # useful for debugging

    def apply(self, data):
        print(data)
        return data

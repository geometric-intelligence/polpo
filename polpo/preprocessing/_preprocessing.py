from joblib import Parallel, delayed
from tqdm import tqdm

from .base import DataLoader, PreprocessingStep


class Pipeline(PreprocessingStep, DataLoader):
    def __init__(self, steps, data=None):
        super().__init__()
        self.steps = steps
        self.data = data

    def apply(self, data=None):
        if self.data is not None:
            data = self.data

        out = data
        for step in self.steps:
            out = step(out)

        return out

    def load(self):
        return self.apply()


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


class IndexSelector(PreprocessingStep):
    def __init__(self, index=0, repeat=False, step=None):
        super().__init__()
        if step is None:
            step = IdentityStep()

        self.index = index
        self.repeat = repeat
        self.step = step

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


class HashWithIncoming(PreprocessingStep):
    def __init__(self, step):
        super().__init__()
        self.step = step

    def apply(self, data):
        new_data = self.step.apply(data)

        return {datum: new_datum for datum, new_datum in zip(data, new_data)}


class Hash(PreprocessingStep):
    def __init__(self, key_index=0):
        super().__init__()
        self.key_index = key_index

    def apply(self, data):
        new_data = {}
        for datum in data:
            if not isinstance(datum, list):
                datum = list(datum)

            key = datum.pop(self.key_index)

            if len(datum) == 1:
                datum = datum[0]

            new_data[key] = datum

        return new_data


class TupleWith(PreprocessingStep):
    def __init__(self, step, incoming_first=True):
        super().__init__()
        self.step = step
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
    def __init__(self, func):
        super().__init__()
        self.func = func

    def apply(self, data):
        return list(filter(self.func, data))


class EmptyRemover(Filter):
    def __init__(self):
        super().__init__(func=lambda x: len(x) > 0)


class NoneRemover(Filter):
    def __init__(self):
        super().__init__(func=lambda x: x is not None)


class ToList(PreprocessingStep):
    # TODO: better naming?
    def apply(self, data):
        return [data]


class DictToValues(PreprocessingStep):
    # TODO: better naming
    def apply(self, data):
        return list(data.values())


class SerialMap(PreprocessingStep):
    def __init__(self, step, pbar=False):
        super().__init__()
        self.step = step
        self.pbar = pbar

    def apply(self, data):
        return [self.step(datum) for datum in tqdm(data, disable=not self.pbar)]


class ParallelMap(PreprocessingStep):
    def __init__(self, step, n_jobs=-1, verbose=0):
        super().__init__()
        self.step = step
        self.n_jobs = n_jobs
        self.verbose = verbose

    def apply(self, data):
        with Parallel(n_jobs=self.n_jobs, verbose=self.verbose) as parallel:
            res = parallel(delayed(self.step)(datum) for datum in data)

        return list(res)


class Map:
    def __new__(cls, step, n_jobs=0, verbose=0):
        if n_jobs != 0:
            return ParallelMap(step, n_jobs=n_jobs, verbose=verbose)

        return SerialMap(step, pbar=verbose > 0)


class MapPipeline:
    # syntax sugar, as it is used often
    def __new__(cls, steps, n_jobs=0, verbose=0):
        if len(steps) == 1:
            step = steps[0]
        else:
            step = Pipeline(steps)
        return Map(step, n_jobs, verbose)


class HashMap(PreprocessingStep):
    def __init__(self, step, pbar=False):
        super().__init__()
        self.step = step
        self.pbar = pbar

    def apply(self, data):
        out = {}
        for key, datum in tqdm(data.items(), disable=not self.pbar):
            out[key] = self.step.apply(datum)

        return out


class IndexMap(PreprocessingStep):
    # TODO: name is confusing
    def __init__(self, step, index=0):
        super().__init__()
        self.step = step
        self.index = index

    def apply(self, data):
        data[self.index] = self.step.apply(data[self.index])

        return data


class Truncater(PreprocessingStep):
    # useful for debugging

    def __init__(self, value):
        super().__init__()
        self.value = value

    def apply(self, data):
        return data[: self.value]

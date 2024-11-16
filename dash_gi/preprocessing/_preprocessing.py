from joblib import Parallel, delayed
from tqdm import tqdm

from .base import DataLoader, PreprocessingStep


class PipelineDataLoader(DataLoader):
    # TODO: accept multiple pipelines? e.g. store info
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def load(self):
        return self.pipeline.apply()


class Pipeline(PreprocessingStep):
    def __init__(self, steps):
        super().__init__()
        self.steps = steps

    def apply(self, data=None):
        out = data
        for step in self.steps:
            out = step.apply(out)

        return out


class ParallelPipeline(PreprocessingStep):
    def __init__(self, pipelines):
        super().__init__()
        self.pipelines = pipelines

    def apply(self, data):
        out = []
        for pipeline in self.pipelines:
            out.append(pipeline.apply(data))

        return list(zip(*out))


class IndexSelector(PreprocessingStep):
    def __init__(self, index, repeat=False):
        super().__init__()
        self.index = index
        self.repeat = repeat

    def apply(self, data):
        selected = data[self.index]
        if self.repeat:
            return [selected] * len(data)

        return selected


class HashWithIncoming(PreprocessingStep):
    def __init__(self, step):
        super().__init__()
        self.step = step

    def apply(self, data):
        new_data = self.step.apply(data)

        return {datum: new_datum for datum, new_datum in zip(data, new_data)}


class TupleWithIncoming(PreprocessingStep):
    def __init__(self, step):
        super().__init__()
        self.step = step

    def apply(self, data):
        new_data = self.step.apply(data)
        return [(datum, new_datum) for datum, new_datum in zip(data, new_data)]


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


class SerialMap(PreprocessingStep):
    def __init__(self, step, pbar=False):
        super().__init__()
        self.step = step
        self.pbar = pbar

    def apply(self, data):
        return [self.step.apply(datum) for datum in tqdm(data, disable=not self.pbar)]


class ParallelMap(PreprocessingStep):
    def __init__(self, step, n_jobs=-1, verbose=0):
        super().__init__()
        self.step = step
        self.n_jobs = n_jobs
        self.verbose = verbose

    def apply(self, data):
        with Parallel(n_jobs=self.n_jobs, verbose=self.verbose) as parallel:
            res = parallel(delayed(self.step.apply)(datum) for datum in data)

        return list(res)


class Map:
    def __new__(cls, step, n_jobs=0, verbose=0):
        if n_jobs != 0:
            return ParallelMap(step, n_jobs=n_jobs, verbose=verbose)

        return SerialMap(step, pbar=verbose > 0)


class Truncater(PreprocessingStep):
    # useful for debugging

    def __init__(self, value):
        super().__init__()
        self.value = value

    def apply(self, data):
        return data[: self.value]


# TODO: add MergeByHash

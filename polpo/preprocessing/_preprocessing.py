import abc
import warnings

from joblib import Parallel, delayed
from tqdm import tqdm

from polpo.utils import is_non_string_iterable, unnest

from .base import Pipeline, PreprocessingStep


def _wrap_step(step=None):
    if step is None:
        step = IdentityStep()

    if is_non_string_iterable(step):
        step = Pipeline(step)

    return step


class StepWrappingPreprocessingStep(PreprocessingStep, abc.ABC):
    def __init__(self, step=None):
        super().__init__()

        self.step = _wrap_step(step)


class ExceptionToWarning(StepWrappingPreprocessingStep):
    def __init__(self, step, warn=True):
        super().__init__(step)
        self.warn = warn

    def apply(self, data):
        try:
            return self.step(data)
        except Exception as e:
            if self.warn:
                warnings.warn(str(e))

        return None


class BranchingPipeline(PreprocessingStep):
    def __init__(self, branches, merger=None):
        if merger is None:
            merger = NestingSwapper()

        super().__init__()
        self.branches = branches
        self.merger = _wrap_step(merger)

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


class IndexSelector(StepWrappingPreprocessingStep):
    def __init__(self, index=0, repeat=False, step=None):
        super().__init__(step)

        self.index = index
        self.repeat = repeat

    def apply(self, data):
        selected = self.step(data[self.index])
        if self.repeat:
            return [selected] * len(data)

        return selected


class RemoveIndex(PreprocessingStep):
    def __init__(self, index=0, inplace=False):
        super().__init__()
        self.index = index
        self.inplace = inplace

    def apply(self, data):
        if not self.inplace:
            data = data.copy()

        data.pop(self.index)
        return data


class Unnest(PreprocessingStep):
    def apply(self, data):
        return unnest(data)


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

class TupleToList(PreprocessingStep):
    def apply(self, data):
        return [x for x in data]


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
        if not isinstance(data, list):
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


class IndexMap(StepWrappingPreprocessingStep):
    # TODO: name is confusing
    def __init__(self, step, index=0):
        super().__init__(step)
        self.index = index

    def apply(self, data):
        data[self.index] = self.step(data[self.index])

        return data


class Prefix(PreprocessingStep):
    def __init__(self, prefix):
        self.prefix = prefix

    def apply(self, value):
        return self.prefix + value


class Truncater(PreprocessingStep):
    # useful for debugging

    def __init__(self, value):
        super().__init__()
        self.value = value

    def apply(self, data):
        if self.value is None:
            return data
        return data[: self.value]


class DataPrinter(PreprocessingStep):
    # useful for debugging

    def __init__(self, silent=False):
        # avoids having to comment code out
        self.silent = silent

    def apply(self, data):
        if not self.silent:
            print(data)
        return data


class PartiallyInitializedStep(PreprocessingStep):
    def __init__(self, Step, **kwargs):
        self.Step = Step
        self.kwargs = kwargs

    def apply(self, data):
        kwargs = self.kwargs.copy()
        dependent_keys = list(filter(lambda x: x.startswith("_"), kwargs.keys()))
        for key in dependent_keys:
            kwargs[key[1:]] = kwargs.pop(key)(data)

        return self.Step(**kwargs)(data)


class IfCondition(StepWrappingPreprocessingStep):
    def __init__(self, step, else_step, condition):
        super().__init__(step)
        self.else_step = _wrap_step(else_step)
        self.condition = condition

    def apply(self, data):
        new_data = self.step(data)

        if self.condition(new_data):
            return self.else_step(data)

        return new_data


class IfEmpty(IfCondition):
    def __init__(self, step, else_step):
        super().__init__(step, else_step, condition=lambda x: len(x) == 0)


class Lambda(PreprocessingStep):
    def __init__(self, string):
        super().__init__()
        self.lambda_ = eval(string)

    def apply(self, data):
        return self.lambda_(data)

import abc
import importlib
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

    def __call__(self, data):
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

    def __call__(self, data):
        out = []
        for pipeline in self.branches:
            out.append(pipeline(data))

        return self.merger(out)


class IdentityStep(PreprocessingStep):
    def __call__(self, data=None):
        return data


class NestingSwapper(PreprocessingStep):
    def __call__(self, data):
        return list(zip(*data))


class IndexSelector(StepWrappingPreprocessingStep):
    def __init__(self, index=0, repeat=False, step=None):
        super().__init__(step)

        self.index = index
        self.repeat = repeat

    def __call__(self, data):
        selected = self.step(data[self.index])
        if self.repeat:
            return [selected] * len(data)

        return selected


class RemoveIndex(PreprocessingStep):
    def __init__(self, index=0, inplace=False):
        super().__init__()
        self.index = index
        self.inplace = inplace

    def __call__(self, data):
        if not self.inplace:
            data = data.copy()

        data.pop(self.index)
        return data


class Unnest(PreprocessingStep):
    def __call__(self, data):
        return unnest(data)


class ListSqueeze(PreprocessingStep):
    def __init__(self, raise_=True):
        self.raise_ = raise_

    def __call__(self, data):
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

    def __call__(self, data):
        new_data = self.step(data)
        ordering = (data, new_data) if self.incoming_first else (new_data, data)
        return [(datum, datum_) for datum, datum_ in zip(*ordering)]


class TupleWithIncoming(TupleWith):
    def __init__(self, step):
        super().__init__(step, incoming_first=True)


class Sorter(PreprocessingStep):
    def __init__(self, key=None):
        super().__init__()
        self.key = key

    def __call__(self, data):
        return sorted(data, key=self.key)


class Filter(PreprocessingStep):
    def __init__(self, func, collection_type=list, to_iter=None):
        if to_iter is None:
            to_iter = lambda x: x

        super().__init__()
        self.func = func
        self.collection_type = collection_type
        self.to_iter = to_iter

    def __call__(self, data):
        return self.collection_type(filter(self.func, self.to_iter(data)))


class EmptyRemover(Filter):
    def __init__(self):
        super().__init__(func=lambda x: len(x) > 0)


class NoneRemover(Filter):
    def __init__(self):
        super().__init__(func=lambda x: x is not None)


class NoneSkipper(StepWrappingPreprocessingStep):
    def __call__(self, data):
        if data is None:
            return data

        return self.step(data)


class EmptySkipper(StepWrappingPreprocessingStep):
    def __call__(self, data):
        if len(data) == 0:
            return data

        return self.step(data)


class ToList(PreprocessingStep):
    # TODO: better naming?
    def __call__(self, data):
        return [data]


class TupleToList(PreprocessingStep):
    def __call__(self, data):
        return [x for x in data]


class SerialMap(StepWrappingPreprocessingStep):
    def __init__(self, step, pbar=False):
        super().__init__(step)
        self.pbar = pbar

    def __call__(self, data):
        return [self.step(datum) for datum in tqdm(data, disable=not self.pbar)]


class ParallelMap(StepWrappingPreprocessingStep):
    def __init__(self, step, n_jobs=-1, verbose=0):
        super().__init__(step)
        self.n_jobs = n_jobs
        self.verbose = verbose

    def __call__(self, data):
        with Parallel(n_jobs=self.n_jobs, verbose=self.verbose) as parallel:
            res = parallel(delayed(self.step)(datum) for datum in data)

        return list(res)


class DecorateToIterable(StepWrappingPreprocessingStep):
    def __call__(self, data):
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

    def __call__(self, data):
        data[self.index] = self.step(data[self.index])

        return data


class Prefix(PreprocessingStep):
    def __init__(self, prefix):
        self.prefix = prefix

    def __call__(self, value):
        return self.prefix + value


class Truncater(PreprocessingStep):
    # useful for debugging

    def __init__(self, value):
        super().__init__()
        self.value = value

    def __call__(self, data):
        if self.value is None:
            return data
        return data[: self.value]


class DataPrinter(PreprocessingStep):
    # useful for debugging

    def __init__(self, silent=False):
        # avoids having to comment code out
        self.silent = silent

    def __call__(self, data):
        if not self.silent:
            print(data)
        return data


class PartiallyInitializedStep(PreprocessingStep):
    def __init__(self, Step, **kwargs):
        super().__init__()
        self.Step = Step
        self.kwargs = kwargs

    def __call__(self, data):
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

    def __call__(self, data):
        if self.condition(data):
            return self.step(data)

        return self.else_step(data)


class IfEmpty(IfCondition):
    def __init__(self, step, else_step):
        super().__init__(step, else_step, condition=lambda x: len(x) == 0)


class Eval(PreprocessingStep):
    """Evaluate string.

    Parameters
    ----------
    expr : str
        String expression to be evaluated.
    imports : list[str]
        Imports required to evaluate string.
    """

    def __init__(self, expr, imports=()):
        super().__init__()
        self._expr = eval(expr, self._locals_from_imports(imports))

    def _locals_from_imports(self, imports):
        locals_ = {}
        for import_ in imports:
            import_ls = import_.split(".")
            module_name = ".".join(import_ls[:-1])
            obj_name = import_ls[-1]

            if obj_name in locals():
                continue

            locals_[obj_name] = getattr(importlib.import_module(module_name), obj_name)
        return locals_

    def __call__(self, *args, **kwargs):
        return self._expr(*args, **kwargs)


class EvalFromImport(Eval):
    """Evaluate imported function.

    Parameters
    ----------
    import_: str
        Import of function to evaluate.
    """

    def __init__(self, import_):
        super().__init__(expr=import_.split(".")[-1], imports=[import_])


class Lambda(Eval):
    """Evaluate lambda function.

    Syntax sugar for `Eval` where `string` is a lambda function.

    Parameters
    ----------
    args : list[str]
        Arguments of lambda function.
    expr : str
        Expression of lambda function.
    imports : list[str]
        Imports required to evaluate string.
    """

    def __init__(self, args, expr, imports=()):
        args_str = ",".join(args)
        lambda_ = f"lambda {args_str}: {expr}"
        super().__init__(lambda_, imports)


class Constant(PreprocessingStep):
    """Constant.

    Parameters
    ----------
    value : any
        Constant.
    """

    def __init__(self, value):
        super().__init__()
        self.value = value

    def __call__(self, value=None):
        """Apply step.

        Parameters
        ----------
        value : str
            Value.

        Returns
        -------
        constant : any
        """
        return value if value is not None else self.value


class Contains(PreprocessingStep):
    """Check if an item is in a collection.

    Examples include substring in string,
    item in list, key in dict.

    Parameters
    ----------
    item : object
    negate : bool
        Whether to negate predicate.
    """

    def __init__(self, item, negate=False):
        super().__init__()
        self.item = item
        self.negate = negate

    def __call__(self, collection):
        """Apply step.

        Returns
        -------
        membership : bool
            Membership or lack of it (depending on negate).
        """
        out = self.item in collection
        if self.negate:
            return not out

        return out


class ContainsAll(PreprocessingStep):
    """Check if a subset of items in a collection."""

    def __init__(self, items, negate=False):
        super().__init__()
        self.items = items
        self.negate = negate

    def __call__(self, collection):
        """Apply step.

        Returns
        -------
        membership : bool
            Membership or lack of it (depending on negate) for all items.
        """
        out = all([item in collection for item in self.items])
        if self.negate:
            return not out

        return out


class MethodApplier(PreprocessingStep):
    """Applies a named method with preset args and kwargs.

    Parameters
    ----------
    method : str
        Named method.
    """

    def __init__(self, *args, method, **kwargs):
        super().__init__()
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def __call__(self, obj):
        """Apply step.

        Parameters
        ----------
        obj : object

        Returns
        -------
        bool
        """
        return getattr(obj, self.method)(*self.args, **self.kwargs)

import collections
import inspect
import itertools


def unnest_list(ls):
    return list(itertools.chain(*ls))


def unnest(ls):
    if not is_non_string_iterable(ls):
        return [ls]

    data = []
    for datum_ in ls:
        data.extend(unnest(datum_))

    return data


def is_non_string_iterable(obj):
    return isinstance(obj, collections.abc.Iterable) and not isinstance(obj, str)


def params_to_kwargs(obj, ignore=(), renamings=None, ignore_private=False, func=None):
    """Get dict with selected object attributes.

    Parameters
    ----------
    obj : object
        Object with desired attributes.
    ignore : tuple[str]
        Attributes to ignore.
    renamings: dict
        Attribute renamings.
    ignore_private: bool
        Whether to ignore private attributes.
    func : callable
        Function to get signature from. Attributes
        not in the signature are ignored.

    Returns
    -------
    kwargs : dict
    """
    kwargs = obj.__dict__.copy()

    if func is not None:
        params = inspect.signature(func).parameters
        ignore = list(ignore) + [key for key in kwargs if key not in params]

    if ignore:
        for key in ignore:
            kwargs.pop(key)

    if renamings is not None:
        for old_key, new_key in renamings.items():
            kwargs[new_key] = kwargs.pop(old_key)

    if ignore_private:
        private_keys = list(filter(lambda key: key.startswith("_"), kwargs.keys()))
        for key in private_keys:
            kwargs.pop(key)

    return kwargs

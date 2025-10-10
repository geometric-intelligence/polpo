import collections
import glob
import inspect
import itertools
import socket
from pathlib import Path

import requests

import polpo.concurrent as pconcurrent


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


def as_list(data):
    if isinstance(data, list):
        return data

    if is_non_string_iterable(data):
        return list(data)

    return [data]


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


def unnest_dict(nested_dict, sep="/", current_key="", flat_dict=None):
    sep_ = sep if current_key else ""

    if flat_dict is None:
        flat_dict = {}

    for key, value in nested_dict.items():
        new_key = f"{current_key}{sep_}{key}"

        if not isinstance(value, dict):
            flat_dict[new_key] = value
        else:
            flat_dict = unnest_dict(
                value, sep=sep, current_key=new_key, flat_dict=flat_dict
            )

    return flat_dict


def nest_dict_inner_level(flat_dict, sep="/"):
    nested_dict = {}

    for key, value in flat_dict.items():
        outer_key, inner_key = key.rsplit(sep, maxsplit=1)

        inner_dict = nested_dict[outer_key] = nested_dict.get(outer_key, {})
        inner_dict[inner_key] = value

    return nested_dict


def nest_dict(flat_dict, sep="/"):
    while True:
        # TODO: make a nicer recursion

        try:
            flat_dict = nest_dict_inner_level(flat_dict, sep=sep)
        except ValueError:
            # when unpack error is raised
            break

    return flat_dict


def extract_unique_key_nested(data):
    if not isinstance(data, dict):
        return data

    if len(data.keys()) == 1:
        return extract_unique_key_nested(data[next(iter(data))])

    return {key: extract_unique_key_nested(value) for key, value in data.items()}


def custom_order(reference):
    # behavior is random if element is not in reference
    order_ = {val: index for index, val in enumerate(reference)}
    n_reference = len(order_)

    def _custom_order(x):
        return order_.get(x, n_reference)

    return _custom_order


def plot_shape_from_n_plots(n_plots, n_axis=2, axis=1):
    # TODO: compute space filler?
    n_axis_0 = min(n_axis, n_plots)
    n_axis_1 = (n_plots + n_axis_0 - 1) // n_axis_0

    if axis == 1:
        return n_axis_1, n_axis_0

    return n_axis_0, n_axis_1


def plot_index_to_shape(index, n_axis, rowise=False):
    # TODO: find better name
    a, b = index // n_axis, index % n_axis

    if rowise:
        return b, a

    return a, b


def get_first(data):
    if isinstance(data, dict):
        return next(iter(data.values()))

    return data[0]


def in_frank():
    return socket.gethostname() == "frank"


def expand_path_names(names):
    out = []
    for name in names:
        if any(ch in name for ch in "*?[]"):
            out.extend(Path(p) for p in glob.glob(name, recursive=True))
        else:
            out.append(Path(name))

    seen = set()
    uniq = []
    for path in out:
        rp = path.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(path)
    return uniq


def is_link_ok(url):
    return requests.get(
        url,
        allow_redirects=True,
        headers={"User-Agent": "link-checker"},
    ).ok


def are_links_ok(urls, workers=16):
    return pconcurrent.thread_map(is_link_ok, urls, workers=workers)

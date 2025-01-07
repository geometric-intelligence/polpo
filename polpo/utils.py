import collections
import itertools


def unnest_list(ls):
    return list(itertools.chain(*ls))


def is_non_string_iterable(obj):
    return isinstance(obj, collections.abc.Iterable) and not isinstance(obj, str)

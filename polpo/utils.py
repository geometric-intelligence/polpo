import itertools


def unnest_list(ls):
    return list(itertools.chain(*ls))

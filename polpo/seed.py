import time


def resolve_seed(seed):
    if seed is None:
        return time.time_ns() % (2**31 - 1)
    return seed

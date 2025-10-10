from concurrent.futures import ThreadPoolExecutor


def thread_map(func, inputs, workers=16):
    n = len(inputs)
    if n == 0:
        return []
    if n == 1:
        return [func(inputs[0])]

    workers = min(workers, n)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(func, inputs))

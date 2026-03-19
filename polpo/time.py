import copy
import time
from contextlib import contextmanager


class Timer:
    def __init__(self):
        self.events = {}

    def start(self, key):
        if key in self.events and "start" in self.events[key]:
            raise RuntimeError(f"{key} already started")

        self.events.setdefault(key, {})["start"] = time.perf_counter()

    def stop(self, key):
        if key not in self.events or "start" not in self.events[key]:
            raise RuntimeError(f"{key} not started")

        event = self.events[key]
        if "end" in event:
            raise RuntimeError(f"{key} already stopped")

        event["end"] = time.perf_counter()
        event["duration"] = event["end"] - event["start"]

    def duration(self, key):
        if key not in self.events:
            raise RuntimeError(f"{key} unknown")

        event = self.events[key]
        if "end" not in event:
            raise RuntimeError(f"{key} not stopped")

        return event["duration"]

    def as_dict(self):
        return copy.deepcopy(self.events)

    @contextmanager
    def __call__(self, key):
        # use e.g. ```with timer("simulation"):```
        self.start(key)
        try:
            yield
        finally:
            self.stop(key)

    def reset(self):
        self.events = {}

        return self

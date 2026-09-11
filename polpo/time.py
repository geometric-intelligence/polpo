import copy
import time
from contextlib import contextmanager
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class Timer:
    def __init__(self):
        self.reset()

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
        dict_ = {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        dict_.update(copy.deepcopy(self.events))
        return dict_

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
        self.started_at = None
        self.finished_at = None
        return self

    def start_run(self):
        self.reset()
        self.started_at = utc_now()
        return self

    def stop_run(self):
        self.finished_at = utc_now()
        return self

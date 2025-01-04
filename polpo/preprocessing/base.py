import abc
import logging
import os

# TODO: is there any difference between a DataLoader and a step?


class DataLoader(abc.ABC):
    """Data loader."""

    @abc.abstractmethod
    def load(self):
        """Load data."""
        pass


class CacheableDataLoader(DataLoader, abc.ABC):
    def __init__(self, use_cache=True):
        self.use_cache = use_cache

    def exists(self, path):
        if self.use_cache and os.path.exists(path):
            logging.info(
                f"Data has already been downloaded... using cached file ('{path}')."
            )
            return True

        return False


class PreprocessingStep(abc.ABC):
    """Preprocessing step."""

    @abc.abstractmethod
    def apply(self, data):
        """Apply step."""
        # takes one argument; name is irrelevant

    def __call__(self, data=None):
        """Apply step."""
        return self.apply(data)

    def __add__(self, other):
        if isinstance(other, list):
            other = Pipeline(other)
        return Pipeline([self, other])


class Pipeline(PreprocessingStep, DataLoader):
    def __init__(self, steps, data=None):
        super().__init__()
        self.steps = steps
        self.data = data

    def apply(self, data=None):
        if self.data is not None:
            data = self.data

        out = data
        for step in self.steps:
            out = step(out)

        return out

    def load(self):
        return self.apply()

    def __add__(self, other):
        steps = self.steps.copy()
        if hasattr(other, "steps"):
            steps.extend(other.steps)
        elif isinstance(other, list):
            steps.extend(other)
        else:
            steps.append(other)

        return Pipeline(steps)

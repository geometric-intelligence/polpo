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

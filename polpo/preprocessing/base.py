import abc

# TODO: is there any difference between a DataLoader and a step?


class DataLoader(abc.ABC):
    """Data loader."""

    @abc.abstractmethod
    def load(self):
        """Load data."""
        pass


class PreprocessingStep(abc.ABC):
    """Preprocessing step."""

    @abc.abstractmethod
    def apply(self, data):
        """Apply step."""
        # takes one argument; name is irrelevant

    def __call__(self, data=None):
        """Apply step."""
        return self.apply(data)

import abc

# TODO: is there any difference between a DataLoader and a step?


class DataLoader(abc.ABC):
    @abc.abstractmethod
    def load(self):
        pass


class PreprocessingStep(abc.ABC):
    @abc.abstractmethod
    def apply(self, data):
        # takes one argument; name is irrelevant
        pass

    def __call__(self, data=None):
        return self.load(data=data)

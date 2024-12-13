import abc


class Plotter(abc.ABC):
    @abc.abstractmethod
    def plot(self, data):
        pass

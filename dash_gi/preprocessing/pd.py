import pandas as pd

from .base import PreprocessingStep


class PdCsvReader(PreprocessingStep):
    def __init__(self, delimiter=","):
        super().__init__()
        self.delimiter = delimiter

    def apply(self, data):
        return pd.read_csv(data, delimiter=",")

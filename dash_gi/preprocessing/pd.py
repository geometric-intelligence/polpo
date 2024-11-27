"""Preprocessing steps involving pandas dataframes."""

import pandas as pd

from .base import PreprocessingStep


class CsvReader(PreprocessingStep):
    """Read csv with pandas.

    Parameters
    ----------
    path : str
        Path.
    delimiter : str
        csv delimiter.
    """

    def __init__(self, path=None, delimiter=","):
        super().__init__()
        self.path = path
        self.delimiter = delimiter

    def apply(self, path=None):
        """Apply step."""
        path = self.path or path
        return pd.read_csv(path, delimiter=",")


class ColumnSelector(PreprocessingStep):
    def __init__(self, column_name):
        self.column_name = column_name

    def apply(self, df):
        return df[self.column_name]


class ColumnsSelector(PreprocessingStep):
    def __init__(self, column_names):
        super().__init__()
        self.column_names = column_names

    def apply(self, df):
        return df[self.column_names]


class Dropna(PreprocessingStep):
    def __init__(self, inplace=True):
        self.inplace = True

    def apply(self, df):
        df.dropna(inplace=self.inplace)
        return df


class DfToDict(PreprocessingStep):
    def __init__(self, orient="dict"):
        self.orient = orient

    def apply(self, df):
        return df.to_dict(self.orient)


class SeriesToDict(PreprocessingStep):
    def apply(self, series):
        return series.to_dict()

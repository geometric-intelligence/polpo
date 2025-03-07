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
        path = path or self.path
        return pd.read_csv(path, delimiter=",")


class ColumnsSelector(PreprocessingStep):
    """Select dataframe column.

    Parameters
    ----------
    column_names : str or list[str]
        Column(s) to select.
    """

    def __init__(self, column_names):
        super().__init__()
        self.column_names = column_names

    def apply(self, df):
        """Apply step.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe.
        """
        return df[self.column_names]


class Dropna(PreprocessingStep):
    """Drop nan in a dataframe.

    Check out
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html.

    Parameters
    ----------
    inplace : bool
        Whether to perform operation in place.
    """

    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def apply(self, df):
        """Apply step.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe.
        """
        out = df.dropna(inplace=self.inplace)
        return out or df


class Drop(PreprocessingStep):
    """Drop specified labels from rows or columns.

    Check out
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html.

    Parameters
    ----------
    inplace : bool
        Whether to perform operation in place.

    """

    def __init__(self, labels, inplace=True):
        super().__init__()
        self.labels = labels
        self.inplace = inplace

    def apply(self, df):
        """Apply step.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe.
        """
        out = df.drop(self.labels, inplace=self.inplace)
        return out or df


class DfToDict(PreprocessingStep):
    """Convert dataframe into dict.

    Check out
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_dict.html.

    Parameters
    ----------
    orient : str
        Determines the type of the values of the dictionary.
    """

    def __init__(self, orient="dict"):
        super().__init__()
        self.orient = orient

    def apply(self, df):
        """Apply step.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe.
        """
        return df.to_dict(self.orient)


class DfCopy(PreprocessingStep):
    """Make a copy of a dataframe.

    Check out
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.copy.html.

    Parameters
    ----------
    deep : bool
        Whether to make a deep copy.
    """

    def __init__(self, deep=True):
        self.deep = deep

    def apply(self, df):
        """Apply step.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe.
        """
        return df.copy(deep=self.deep)


class UpdateColumnValues(PreprocessingStep):
    """Update dataframe column values.

    Operation is done in place.
    Use ``DfCopy`` if that is not desired.

    Parameters
    ----------
    column_name : str
        Column to be modified
    func : callable
        Function to apply to each element.
    """

    def __init__(self, column_name, func):
        super().__init__()
        self.column_name = column_name
        self.func = func

    def apply(self, df):
        """Apply step.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe.
        """
        df[self.column_name] = df[self.column_name].apply(self.func)

        return df


class SeriesToDict(PreprocessingStep):
    """Convert series into dict.

    Check out
    https://pandas.pydata.org/docs/reference/api/pandas.Series.to_dict.html.
    """

    def apply(self, series):
        """Apply step.

        Parameters
        ----------
        series : pandas.Series
            Series.
        """
        return series.to_dict()


class IndexSetter(PreprocessingStep):
    """Set the DataFrame index using existing columns.

    Check out
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.set_index.html.

    Parameters
    ----------
    key : str
        New index.
    inplace : bool
        Whether to perform operation in place.
    drop : bool
        Delete columns to be used as the new index.
    verify_integrity : bool
        Check the new index for duplicates.
    """

    def __init__(self, key, inplace=True, drop=False, verify_integrity=True):
        self.key = key
        self.inplace = inplace
        self.drop = drop
        self.verify_integrity = verify_integrity

    def apply(self, df):
        """Apply step.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe.
        """
        out = df.set_index(
            self.key,
            inplace=self.inplace,
            drop=self.drop,
            verify_integrity=self.verify_integrity,
        )
        return out or df

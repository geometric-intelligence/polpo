"""Preprocessing steps involving pandas dataframes."""

import pandas as pd

from polpo.utils import params_to_kwargs

from .base import Pipeline, PreprocessingStep


class CsvReader(PreprocessingStep):
    """Read csv with pandas.

    https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

    Parameters
    ----------
    path : str
        Path.
    delimiter : str
        csv delimiter.
    nrows: int
        Number of rows of file to read. Useful for reading pieces of large files.

    """

    def __init__(self, path=None, delimiter=",", nrows=None):
        super().__init__()
        self.path = path
        self.delimiter = delimiter
        self.nrows = nrows

    def __call__(self, path=None):
        """Apply step."""
        path = path or self.path
        return pd.read_csv(
            path,
            **params_to_kwargs(self, ignore=("path",)),
        )


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

    def __call__(self, df):
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

    def __call__(self, df):
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
    axis : {0 or ‘index’, 1 or ‘columns’}
        Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
    """

    def __init__(self, labels, axis=0, inplace=True):
        super().__init__()
        self.labels = labels
        self.inplace = inplace
        self.axis = axis

    def __call__(self, df):
        """Apply step.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe.
        """
        out = df.drop(self.labels, inplace=self.inplace, axis=self.axis)
        return out if out is not None else df


class DfToDict(PreprocessingStep):
    """Convert dataframe into dict.

    Check out
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_dict.html.

    Parameters
    ----------
    orient : str
        Determines the type of the values of the dict.
    """

    def __init__(self, orient="dict"):
        super().__init__()
        self.orient = orient

    def __call__(self, df):
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

    def __call__(self, df):
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

    def __call__(self, df):
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

    def __call__(self, series):
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

    def __call__(self, df):
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


class DfIsInFilter(PreprocessingStep):
    def __init__(self, column_name, values, negate=False, readonly=True):
        # NB: readonly to avoid pd.errors.SettingWithCopyWarning
        super().__init__()
        self.column_name = column_name
        self.values = values
        self.negate = negate
        self.readonly = readonly

    def __call__(self, df):
        indices = df[self.column_name].isin(self.values)
        if self.negate:
            indices = ~indices

        out = df[indices]

        if self.readonly:
            return out

        return out.copy()


class DfFilter(PreprocessingStep):
    def __init__(self, filter_, negate=False):
        super().__init__()
        self.filter = filter_
        self.negate = negate

    def __call__(self, df):
        indices = self.filter(df)
        if self.negate:
            indices = ~indices

        return df[indices]


class ColumnToDict(Pipeline):
    """Extract a column from a dataframe and convert it to a dict.

    Parameters
    ----------
    column_name : str
        Name of the column to extract.
    """

    def __init__(self, column_name):
        super().__init__(
            steps=[
                ColumnsSelector(column_names=column_name),
                SeriesToDict(),
            ]
        )


class ColumnsToDict(Pipeline):
    """
    Extract one or more columns from a dataframe and convert them to a row-indexed dict.

    Parameters
    ----------
    column_names : list of str
        Names of the columns to extract.
    """

    # TODO: make dropna optional?

    def __init__(self, column_names):
        super().__init__(
            steps=[
                ColumnsSelector(column_names=column_names),
                Dropna(),
                DfToDict(orient="index"),
            ]
        )

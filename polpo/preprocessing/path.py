"""Preprocessing steps involving files."""

import os
import warnings

from .base import PreprocessingStep


class FileRule(PreprocessingStep):
    """File rule.

    Parameters
    ----------
    value : str
        Value to call ``func`` on.
    func : callable
        ``str`` function that outputs a boolean.
    """

    def __init__(self, value, func="startswith"):
        super().__init__()
        self.value = value
        self.func = func

    def apply(self, file):
        """Apply step.

        Parameters
        ----------
        file : str

        Returns
        -------
        bool
        """
        func = getattr(file, self.func)
        return func(self.value)


class FileFinder(PreprocessingStep):
    """Find files given rules.

    Parameters
    ----------
    data_dir : str
        Searching directory.
    rules : list[FileRule]
        Rules to filter files with.
    warn : bool
        Whether to warn if can't find file.
    """

    def __init__(self, data_dir=None, rules=(), warn=True):
        super().__init__()
        self.data_dir = data_dir
        self.rules = rules
        self.warn = warn

    def apply(self, data=None):
        """Apply step.

        Parameters
        ----------
        data_dir : str
            Searching directory.

        Returns
        -------
        list[str] or str
        """
        data_dir = self.data_dir or data

        files = os.listdir(data_dir)

        # TODO: also implement as a pipeline?
        for rule in self.rules:
            files = filter(rule.apply, files)

        out = list(map(lambda name: os.path.join(data_dir, name), files))

        if self.warn and len(out) == 0:
            warnings.warn(f"Couldn't find file in: {data_dir}")

        if len(out) == 1:
            return out[0]

        return out


class Path(PreprocessingStep):
    """Get a path.

    Parameters
    ----------
    path : str
        Path.
    """

    def __init__(self, path):
        super().__init__()
        self.path = path

    def apply(self, path=None):
        """Apply step.

        Parameters
        ----------
        path : str
            Path.

        Returns
        -------
        str
        """
        return self.path or path


class PathShortener(PreprocessingStep):
    """Shorten path.

    Parameters
    ----------
    init_index : int
        Initial index for concatenation.
    last_index : int
        Last index for concatenation.
    """

    def __init__(self, init_index=-1, last_index=None):
        self.init_index = init_index
        self.last_index = last_index

    def apply(self, path):
        """Apply step.

        Parameters
        ----------
        path : str
            Path name.

        Returns
        -------
        str
        """
        path_ls = path.split(os.path.sep)
        return f"{os.path.sep}".join(path_ls[self.init_index : self.last_index])

"""Preprocessing steps involving strings."""

import re

from .base import PreprocessingStep


class DigitFinder(PreprocessingStep):
    """Find digits in a strings.

    Parameters
    ----------
    index : int
        Which index to pick in case of multiple
        digit sets are expected to be found.
    """

    def __init__(self, index=-1):
        self.index = index

    def apply(self, string):
        """Apply step.

        Parameters
        ----------
        string : str

        Returns
        -------
        int
        """
        digits = re.findall(r"\d+", string)
        return int(digits[self.index])

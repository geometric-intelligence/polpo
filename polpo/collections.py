import pandas as pd


def swap_nested_dict(nested_dict):
    """Swap nested levels.

    Parameters
    ----------
    nested_dict : dict

    Returns
    -------
    dict
    """
    return pd.DataFrame(nested_dict).transpose().to_dict()

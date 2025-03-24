from tqdm import tqdm

from polpo.collections import swap_nested_dict

from ._preprocessing import (
    Filter,
    StepWrappingPreprocessingStep,
    _wrap_step,
)
from .base import PreprocessingStep


class Hash(PreprocessingStep):
    def __init__(self, key_index=0, ignore_none=False, ignore_empty=False):
        super().__init__()
        self.key_index = key_index
        self.ignore_none = ignore_none
        self.ignore_empty = ignore_empty

    def apply(self, data):
        new_data = {}
        for datum in data:
            if not isinstance(datum, list):
                datum = list(datum)

            key = datum.pop(self.key_index)

            if len(datum) == 1:
                datum = datum[0]

            if (self.ignore_none and datum is None) or (
                self.ignore_empty and isinstance(datum, list) and len(datum) == 0
            ):
                continue

            new_data[key] = datum

        return new_data


class DictMerger(PreprocessingStep):
    # NB: not shared keys are ignored

    def _collect_shared_keys(self, data):
        keys = set(data[0].keys())
        for datum in data[1:]:
            keys = keys.intersection(set(datum.keys()))

        return keys

    def apply(self, data):
        shared_keys = self._collect_shared_keys(data)
        out = []
        for key in shared_keys:
            out.append([datum[key] for datum in data])

        return out


class HashWithIncoming(StepWrappingPreprocessingStep):
    def __init__(self, step=None, key_step=None):
        super().__init__(step)
        self.key_step = _wrap_step(key_step)

    def apply(self, keys_data):
        values_data = self.step(keys_data)
        keys_data = self.key_step(keys_data)

        return {key: value for key, value in zip(keys_data, values_data)}


class DictFilter(Filter):
    def __init__(self, func, filter_keys=False):
        if filter_keys:
            func_ = lambda x: func(x[0])
        else:
            func_ = lambda x: func(x[1])
        super().__init__(func_, collection_type=dict, to_iter=lambda x: x.items())


class DictKeysFilter(DictFilter):
    # TODO: update and/or rename/deletes
    # TODO: aka SelectKeySubset
    def __init__(self, values, keep=True):
        if keep:
            func = lambda key: key in values
        else:
            func = lambda key: key not in values

        super().__init__(func, filter_keys=True)


class DictNoneRemover(Filter):
    def __init__(self):
        super().__init__(
            func=lambda x: x[1] is not None,
            collection_type=dict,
            to_iter=lambda x: x.items(),
        )

class DictExtractKey(PreprocessingStep):
    def __init__(self, data, key):
        self.key = key
        self.data = data

    def apply(self, data):
        return self.data[self.key]


class DictToValuesList(PreprocessingStep):
    def apply(self, data):
        return list(data.values())


class DictToTuplesList(PreprocessingStep):
    def apply(self, data):
        return list(zip(data.keys(), data.values()))


class OutputDict(PreprocessingStep):
    def __init__(self, key):
        self.key = key

    def apply(self, data):
        if isinstance(data, dict):  # If already a dictionary, add the new key
            data[self.key] = data
            return data
        else:
            return {self.key: data}

class DictAddMesh(PreprocessingStep):
    """
    Adds a Trimesh object to an existing dictionary under a specified key.
    """

    def __init__(self, key="mesh", data=None):
        """
        Parameters:
        - key (str): The dictionary key under which to store the Trimesh object (default: "mesh").
        """
        self.key = key
        self.data = data

    def apply(self, mesh):
        """
        Adds `mesh` to `data` under the given key.

        Parameters:
        - data (dict): The existing dictionary (e.g., template_dict).
        - mesh (Trimesh): The mesh object to add.

        Returns:
        - dict: The updated dictionary.
        """
        if not isinstance(self.data, dict):
            raise ValueError("DictAddMesh expects a dictionary as input.")

        self.data[self.key] = mesh  # Add mesh to dictionary
        return self.data  # Return updated dictionary

class DictAddTemplateMesh(PreprocessingStep):
    def __init__(self, key="template", data=None):
        """
        Parameters:
        - key (str): The dictionary key under which to store the Trimesh object (default: "mesh").
        """
        self.key = key
        self.data = data

    def apply(self, data, image):
        """
        Adds `mesh` to `data` under the given key.

        Parameters:
        - data (dict): The existing dictionary (e.g., template_dict).
        - mesh (Trimesh): The mesh object to add.

        Returns:
        - dict: The updated dictionary.
        """
        if not isinstance(self.data, dict):
            raise ValueError("DictAddMesh expects a dictionary as input.")

        self.data[self.key] = image["mesh"]  # Add mesh to dictionary
        return self.data  # Return updated dictionary

class DictFromTuple(PreprocessingStep):
    def apply(self, keys, tup):
        """keys is a list of keys
            data is a tuple of values"""
        new_data = {}
        for key, data in zip(keys, tup):
            new_data[key] = data
        return new_data


class DictUpdate(PreprocessingStep):
    def apply(self, data):
        new_data = data[0].copy()
        for datum in data[1:]:
            new_data.update(datum)

        return new_data

class DictAddEntry(PreprocessingStep):
    """
    A preprocessing step that adds an entry to a dictionary.
    """

    def __init__(self, data, key, value):
        self.data = data
        self.key = key
        self.value = value

    def apply(self, data):
        if not isinstance(self.data, dict):
            raise ValueError("DictAddEntry expects a dictionary as input.")
        
        self.data[self.key] = self.value
        return self.data


class DictRemoveEntry(PreprocessingStep):
    def apply(self, data, key):
        new_data = data.copy()
        # new_data.pop(self.key, None)
        del new_data[key]
        return new_data


class DictMap(StepWrappingPreprocessingStep):
    def __init__(
        self, step=None, pbar=False, key_step=None, special_keys=(), special_step=None
    ):
        super().__init__(step)
        self.pbar = pbar
        self.key_step = _wrap_step(key_step)
        self.special_keys = special_keys
        self.special_step = _wrap_step(special_step)

    def apply(self, data):
        out = {}
        for key, datum in tqdm(data.items(), disable=not self.pbar):
            if key in self.special_keys:
                value = self.special_step(datum)
            else:
                value = self.step(datum)
            out[self.key_step(key)] = value

        return out


class NestedDictSwapper(PreprocessingStep):
    def apply(self, nested_dict):
        return swap_nested_dict(nested_dict)

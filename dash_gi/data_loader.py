import abc
import os
import warnings

import nibabel as nib
import pandas as pd
from tqdm import tqdm

# TODO: rename to data_processing? maybe have a folder?


class DataLoader(abc.ABC):
    @abc.abstractmethod
    def load(self):
        pass


class PipelineDataLoader(DataLoader):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def load(self):
        return self.pipeline.apply()


class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def apply(self):
        out = None
        for step in self.steps:
            # TODO: add progress bar in for some cases?
            out = step.apply(out)

        return out


class FileRule:
    def __init__(self, value, func="startswith"):
        self.value = value
        self.func = func

    def apply(self, file):
        func = getattr(file, self.func)
        return func(self.value)


class FileFinder:
    def __init__(self, data_dir=None, rules=(), warn=True):
        self.data_dir = data_dir
        self.rules = rules
        self.warn = warn

    def apply(self, data=None):
        data_dir = data or self.data_dir

        files = os.listdir(data_dir)
        for rule in self.rules:
            files = filter(rule.apply, files)

        out = list(map(lambda name: os.path.join(data_dir, name), files))

        if self.warn and len(out) == 0:
            return warnings.warn(f"Couldn't find file in: {data_dir}")

        if len(out) == 1:
            return out[0]

        return out


class Path:
    def __init__(self, path):
        self.path = path

    def apply(self, data=None):
        return data or self.path


class MriImageLoader:
    # TODO: add logger?
    def apply(self, filename):
        img = nib.load(filename)
        img_data = img.get_fdata()

        return img_data


class HashWithIncoming:
    def __init__(self, step):
        self.step = step

    def apply(self, data):
        new_data = self.step.apply(data)

        return {datum: new_datum for datum, new_datum in zip(data, new_data)}


class Sorter:
    def apply(self, data):
        return sorted(data)


class EmptyRemover:
    def apply(self, data):
        return list(filter(lambda x: len(x) > 0, data))


class MapStep:
    def __init__(self, step, pbar=False):
        self.step = step
        self.pbar = pbar

    def apply(self, data):
        return [self.step.apply(datum) for datum in tqdm(data, disable=not self.pbar)]


class Truncater:
    # useful for debugging

    def __init__(self, value):
        self.value = value

    def apply(self, data):
        return data[: self.value]


class PdCsvReader:
    def __init__(self, delimiter=","):
        self.delimiter = delimiter

    def apply(self, data):
        return pd.read_csv(data, delimiter=",")

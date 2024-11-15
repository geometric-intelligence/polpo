import os
import warnings

from .base import PreprocessingStep


class FileRule(PreprocessingStep):
    def __init__(self, value, func="startswith"):
        super().__init__()
        self.value = value
        self.func = func

    def apply(self, file):
        func = getattr(file, self.func)
        return func(self.value)


class FileFinder(PreprocessingStep):
    def __init__(self, data_dir=None, rules=(), warn=True):
        super().__init__()
        self.data_dir = data_dir
        self.rules = rules
        self.warn = warn

    def apply(self, data=None):
        data_dir = data or self.data_dir

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
    def __init__(self, path):
        super().__init__()
        self.path = path

    def apply(self, data=None):
        return data or self.path


class PathShortener(PreprocessingStep):
    def __init__(self, init_index=-2, last_index=-1):
        self.init_index = init_index
        self.last_index = last_index

    def apply(self, path_name):
        path_ls = path_name.split(os.path.sep)
        return f"{os.path.sep}".join(path_ls[self.init_index : self.last_index])

import abc


class TemplateFinder(abc.ABC):
    def __call__(self, subj_id, subj_dataset):
        pass

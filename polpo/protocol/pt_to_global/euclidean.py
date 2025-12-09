import numpy as np

from .base import TemplateFinder


class EuclideanTemplateFinder(TemplateFinder):
    def __init__(self, space):
        super().__init__()
        self.space = space

    def __call__(self, subj_id, subj_dataset):
        # TODO: use FrechetMean properly
        meshes = list(subj_dataset.values())
        metric = self.space.metric
        if hasattr(metric, "diffeo"):
            meshes = metric.diffeo(meshes)

        # TODO: need to implement inverse_map
        template = np.mean(meshes, axis=0)

        return metric.diffeo.inverse(template)


class EuclideanGlobalTemplateFinder(EuclideanTemplateFinder):
    # TODO: think about this
    # implement by composition (same for deformetrica)

    def __call__(self, templates, dataset=None, subj_id="global"):
        return super().__call__(subj_id, templates)

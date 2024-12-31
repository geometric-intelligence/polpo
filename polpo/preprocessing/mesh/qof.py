import numpy as np

from polpo.preprocessing.base import PreprocessingStep


class ColorCentroids(PreprocessingStep):
    def apply(self, data):
        vertices, _, vertex_colors = data

        colors = np.unique(vertex_colors, axis=0)

        centroids = []
        for color in colors:
            color_idx = np.all(vertex_colors == color, axis=-1)

            centroids.append(np.mean(vertices[color_idx], axis=0))

        return np.stack(centroids)

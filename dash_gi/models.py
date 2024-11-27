import abc
import copy

import numpy as np

from .preprocessing import (
    IdentityStep,
    IndexSelector,
    ListSqueeze,
    Map,
    ParallelPipeline,
    Pipeline,
)
from .preprocessing.mesh import FromCombinatorialStructure, ToVertices
from .preprocessing.np import AtLeast2d, Stack, ToArray


class Model(abc.ABC):
    @abc.abstractmethod
    def predict(self, X):
        pass


class SklearnLikeModel(Model, abc.ABC):
    @abc.abstractmethod
    def fit(self, X, y=None):
        pass


class ModelFactory(abc.ABC):
    @abc.abstractmethod
    def create(self):
        # returns a `Model`
        pass


class ConstantOutput(Model):
    # for debugging
    def __init__(self, value):
        super().__init__()
        self.value = value

    def predict(self, X=None):
        return self.value


class ListLookup(Model):
    def __init__(self, data, tar=0):
        super().__init__()
        self.data = data
        self.tar = tar

    def predict(self, X):
        # NB: expects a (int,)
        return self.data[X[0] - self.tar]


class PdDfLookup(Model):
    def __init__(self, df, output_keys, tar=0):
        super().__init__()
        self.df = df
        self.output_keys = output_keys
        self.tar = tar

    def predict(self, X):
        # NB: expects a (int,)

        # TODO: make it input key based instead?
        df_session = self.df.iloc[X[0] - self.tar]
        df_session.columns = self.df.columns

        return [df_session.get(key) for key in self.output_keys]


class MriSlicesLookup(Model):
    def __init__(self, data, index_tar=1, index_ordering=(0, 1, 2)):
        self.data = data
        self.index_tar = index_tar
        self.index_ordering = index_ordering

    def predict(self, X):
        index, *slice_indices = X

        datum = self.data[index - self.index_tar]

        slices = []
        for index, slice_index in zip(self.index_ordering, slice_indices):
            slicing_indices = [slice(None)] * 3
            slicing_indices[index] = slice_index
            slices.append(datum[tuple(slicing_indices)])

        common_width = max([len(slice_[:, 0]) for slice_ in slices])
        common_height = max([len(slice_[0]) for slice_ in slices])

        for i_slice, slice_ in enumerate(slices):
            if len(slice_[:, 0]) < common_width:
                diff = common_width - len(slice_[:, 0])
                slice_ = np.pad(
                    slice_, ((diff // 2, diff // 2), (0, 0)), mode="constant"
                )
                slices[i_slice] = slice_
            if len(slice_[0]) < common_height:
                diff = common_height - len(slice_[0])
                slice_ = np.pad(
                    slice_, ((0, 0), (diff // 2, diff // 2)), mode="constant"
                )
                slices[i_slice] = slice_

        return slices


class VertexBasedMeshRegressor(SklearnLikeModel):
    # TODO: split into pipeline based? check sklearn
    def __init__(
        self, vertex_model, meshes2vertices=None, x2x=None, vertices2meshes=None
    ):
        super().__init__()
        if meshes2vertices is None:
            meshes2vertices = Pipeline(
                steps=[
                    Map(step=Pipeline([ToVertices(), ToArray()])),
                    Stack(),
                ]
            )

        if x2x is None:
            # x2x = Pipeline(steps=[ToArray(), Reshape(shape=(-1, 1))])
            # TODO: update
            x2x = Pipeline(steps=[ToArray(), AtLeast2d()])

        if vertices2meshes is None:
            vertices2meshes = Pipeline(
                steps=[
                    ParallelPipeline(
                        pipelines=[
                            Pipeline([IndexSelector(index=0, repeat=True)]),
                            IndexSelector(index=1),
                        ]
                    ),
                    Map(step=FromCombinatorialStructure()),
                    ListSqueeze(),
                ]
            )

        self.vertex_model = vertex_model
        self.meshes2vertices = meshes2vertices
        self.x2x = x2x
        self.vertices2meshes = vertices2meshes

        self._template_mesh = None

    def fit(self, X, y):
        X = self.x2x(X)
        vertices = self.meshes2vertices(y)
        self._template_mesh = y[0]

        self.vertex_model.fit(X, vertices)

        return self

    def predict(self, X):
        X = self.x2x(X)
        vertices = self.vertex_model.predict(X)

        return self.vertices2meshes((self._template_mesh, vertices))


class SklearnLikeModelFactory(ModelFactory):
    # TODO: inherit from some pipeline based factory?

    def __init__(self, model, data, pipeline=None):
        self.model = model
        self.data = data

        if pipeline is None:
            pipeline = IdentityStep()

        self.pipeline = pipeline

    def create(self):
        X, y = self.pipeline(self.data)
        return self.model.fit(X=X, y=y)

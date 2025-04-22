import abc

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from polpo.plot.mri import MriSlicer
from polpo.preprocessing import IdentityStep, ListSqueeze
from polpo.sklearn.adapter import AdapterPipeline, MapTransformer
from polpo.sklearn.base import GetParamsMixin
from polpo.sklearn.compose import ObjectBasedTransformedTargetRegressor
from polpo.sklearn.dict import BiDictToValuesList
from polpo.sklearn.mesh import BiMeshesToVertices
from polpo.sklearn.np import BiFlattenButFirst, BiHstack
from polpo.sklearn.point_cloud import (
    FittableRegisteredPointCloudSmoothing,
)


def _to_list_with_false(obj):
    return [obj] if obj else []


def _clone_repeat(obj, n_reps):
    if isinstance(obj, (list, tuple)):
        if len(obj) != n_reps:
            raise ValueError(f"Inconsistent size for obj: {n_reps} != {len(obj)}")

        return obj

    if isinstance(obj, sklearn.base.BaseEstimator):
        return [sklearn.base.clone(obj) for _ in range(n_reps)]

    return [obj] * n_reps


class Model(abc.ABC):
    @abc.abstractmethod
    def predict(self, X):
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
    def __init__(self, data, index_tar=1, slicer=None):
        if slicer is None:
            slicer = MriSlicer()

        self.data = data
        self.index_tar = index_tar
        self.slicer = slicer

    @classmethod
    def from_index_ordering(cls, data, index_tar=1, index_ordering=(0, 1, 2)):
        slicer = MriSlicer(index_ordering=index_ordering)
        return cls(data, index_tar, slicer)

    def predict(self, X):
        index, *slice_indices = X

        datum = self.data[index - self.index_tar]
        return self.slicer.slice(datum, slice_indices)


def X2xPipeline(scaler=None, as_pipe=True):
    # useful for object regressor
    # NB: sklearn compatible pipelines
    steps = [np.asarray, np.atleast_2d]

    if scaler is not None:
        steps.append(scaler)

    if as_pipe:
        return AdapterPipeline(steps=steps)

    return steps


def Meshes2FlatVertices(smoother=False, as_pipe=True):
    if smoother is None:
        smoother = FittableRegisteredPointCloudSmoothing(n_neighbors=10)

    steps = (
        [BiMeshesToVertices()]
        + _to_list_with_false(smoother)
        + [
            FunctionTransformer(func=np.stack),
            BiFlattenButFirst(),
        ]
    )

    if as_pipe:
        return AdapterPipeline(steps=steps)

    return steps


def Meshes2Comps(
    scaler=None,
    dim_reduction=None,
    smoother=False,
    mesh_transform=None,
    as_pipe=True,
):
    # mesh_transform (e.g. AffineTransformation)
    if scaler is None:
        scaler = StandardScaler(with_std=False)

    if dim_reduction is None:
        dim_reduction = PCA()

    if smoother is None:
        smoother = FittableRegisteredPointCloudSmoothing(n_neighbors=10)

    if mesh_transform is not None:
        mesh_transform = FunctionTransformer(inverse_func=mesh_transform)

    steps = (
        _to_list_with_false(mesh_transform)
        + [BiMeshesToVertices(), FunctionTransformer(func=np.stack)]
        + _to_list_with_false(smoother)
        + [BiFlattenButFirst()]
        + _to_list_with_false(scaler)
        + [dim_reduction]
    )

    if as_pipe:
        return AdapterPipeline(steps=steps)

    return steps


def DictMeshes2y(pipes, as_pipe=True):
    steps = [
        BiDictToValuesList(),
        MapTransformer(pipes),
        BiHstack(),
    ]
    if as_pipe:
        return AdapterPipeline(steps=steps)

    return steps


def DictMeshes2FlatVertices(n_pipes, smoother=False, as_pipe=True):
    pipes = [
        Meshes2FlatVertices(smoother=smoother_)
        for smoother_ in _clone_repeat(smoother, n_reps=n_pipes)
    ]

    return DictMeshes2y(pipes, as_pipe=as_pipe)


def DictMeshes2Comps(
    n_pipes,
    scaler=None,
    dim_reduction=None,
    smoother=False,
    mesh_transform=None,
    as_pipe=True,
):
    # syntax sugar
    pipes = [
        Meshes2Comps(
            scaler=scaler_,
            dim_reduction=dim_reduction_,
            smoother=smoother_,
            mesh_transform=mesh_transform_,
        )
        for scaler_, dim_reduction_, smoother_, mesh_transform_ in zip(
            _clone_repeat(scaler, n_reps=n_pipes),
            _clone_repeat(dim_reduction, n_reps=n_pipes),
            _clone_repeat(smoother, n_reps=n_pipes),
            _clone_repeat(mesh_transform, n_reps=n_pipes),
        )
    ]

    return DictMeshes2y(pipes, as_pipe=as_pipe)


class ObjectRegressor(GetParamsMixin, SklearnPipeline):
    # just syntax sugar for sklearn.Pipeline

    def __init__(self, model, objs2y, x2x=None):
        # NB: recall sklearn.MultiOutputRegressor
        if model is None:
            model = LinearRegression()

        if x2x is None:
            x2x = X2xPipeline()

        tmodel = ObjectBasedTransformedTargetRegressor(
            regressor=model,
            transformer=objs2y,
            check_inverse=False,
        )

        super().__init__(
            steps=[
                ("pre", x2x),
                ("model", tmodel),
            ]
        )

    @property
    def objs2y(self):
        return self["model"].transformer


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

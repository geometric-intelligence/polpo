from sklearn.preprocessing import FunctionTransformer

from polpo.preprocessing import Map
from polpo.preprocessing.mesh import FromCombinatorialStructure, ToVertices


class InvertibleMeshesToVertices(FunctionTransformer):
    def __init__(self, index=0):
        # index: which mesh to select
        self.index = index

        super().__init__(
            func=Map(ToVertices()),
            inverse_func=Map(FromCombinatorialStructure()),
            check_inverse=False,
        )

    def fit(self, X, y=None):
        # y is ignored
        self.inverse_func.step.mesh = X[self.index]
        return self

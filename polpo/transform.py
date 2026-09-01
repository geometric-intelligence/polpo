# TODO: collect other instances of this

# TODO: connect with geomstats


class InvertibleTransform:
    def __call__(self, x):
        raise NotImplementedError

    def inverse(self, x):
        raise NotImplementedError


class TransformAdapter:
    """Adapt a transform/inverse_transform interface to call/inverse."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform.transform(x)

    def inverse(self, x):
        return self.transform.inverse_transform(x)


def _as_invertible_transform(transform):
    if callable(transform) and hasattr(transform, "inverse"):
        return transform

    if hasattr(transform, "transform") and hasattr(transform, "inverse_transform"):
        return TransformAdapter(transform)

    raise TypeError(
        "Expected an invertible transform or an object defining "
        "'transform' and 'inverse_transform'."
    )


class CompositeTransform(InvertibleTransform):
    def __init__(self, transforms):
        self.transforms = [
            _as_invertible_transform(transform) for transform in transforms
        ]

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

    def inverse(self, x):
        for transform in reversed(self.transforms):
            x = transform.inverse(x)
        return x

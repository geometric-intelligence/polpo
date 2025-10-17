import numpy as np


def kabsch(source, target, as_homogeneous=True):
    # TODO: make geomstats implementation explicit

    # source, target: (N,3) corresponding point clouds
    centroid_A = source.mean(axis=0)
    centroid_B = target.mean(axis=0)

    source_ = source - centroid_A
    target_ = target - centroid_B
    H = source_.T @ target_
    U, S, Vt = np.linalg.svd(H)
    rotation = Vt.T @ U.T

    # reflection fix
    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = Vt.T @ U.T

    translation = centroid_B - rotation @ centroid_A

    if as_homogeneous:
        return np.vstack(
            [
                np.hstack([rotation, translation[:, None]]),
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    return rotation, translation

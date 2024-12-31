import numpy as np
import open3d as o3d

from polpo.preprocessing.base import PreprocessingStep


class O3dPointCloudFromNp(PreprocessingStep):
    def apply(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd


class O3dIcp(PreprocessingStep):
    def __init__(self, threshold=None, estimation_method=None):
        if estimation_method is None:
            estimation_method = (
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
        self.threshold = threshold
        self.estimation_method = estimation_method

    def _threshold(self, source, target):
        return (
            np.amax(
                np.linalg.norm(
                    np.array(source.points) - np.array(target.points),
                    axis=-1,
                )
            )
            * 1.2
        )

    def apply(self, data):
        source, target = data

        threshold = self.threshold or self._threshold(source, target)

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            estimation_method=self.estimation_method,
        )
        return reg_p2p.transformation

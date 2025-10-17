from polpo.preprocessing.base import IdentityStep, Pipeline


class PointCloudAdapter(Pipeline):
    def __init__(
        self,
        template_faces,
        step,
        mesh2points=None,
        points2points=None,
        data2mesh=None,
    ):
        # converters default to pv if None
        # TODO: consider multi

        if mesh2points is None:
            mesh2points = lambda x: x.points

        if points2points is None:
            points2points = IdentityStep()

        if data2mesh is None:
            from polpo.preprocessing.mesh.conversion import PvFromData

            data2mesh = PvFromData()

        super().__init__(
            steps=[
                mesh2points,
                step,
                points2points + (lambda points: (points, template_faces)) + data2mesh,
            ]
        )

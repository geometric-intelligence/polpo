import shutil

from in_out.array_readers_and_writers import write_3D_array

import polpo.deformetrica as pdefo

from .core import DeterministicAtlasDir


class FrechetMean:
    def __init__(self, metric, initial_step_size=1e-4, recompute=False):
        # TODO: space? seems overkill for goal
        self.metric = metric
        self.initial_step_size = initial_step_size

        self.recompute = recompute

        self.estimate_ = None

    def _dir_exists(self, dirname):
        if self.recompute and dirname.exists():
            shutil.rmtree(dirname)

        return dirname.exists()

    def fit(self, X, atlas_id):
        self.estimate_ = None

        dir_ = DeterministicAtlasDir(
            self.metric.dir_config.atlas_dir / atlas_id, points=X
        )

        if not self._dir_exists(dir_.dirname):
            if len(X) > 1:
                dataset = {point.id: point.as_vtk_path() for point in X}
                pdefo.learning.estimate_deterministic_atlas(
                    targets=dataset,
                    output_dir=dir_.dirname,
                    initial_step_size=self.initial_step_size,
                    kernel_width=self.metric.kernel_width,
                    **self.metric.registration_kwargs,
                )

                momenta = pdefo.io.load_momenta(dir_.dirname, as_path=False)
                for momenta_, point in zip(momenta, X):
                    filename = f"DeterministicAtlas__EstimatedParameters__Momenta__subject_{point.id}.txt"
                    write_3D_array(momenta_, dir_.dirname, filename)
            else:
                dir_.write_mesh()

            dir_.write_json()

        self.estimate_ = dir_.template()

        return self

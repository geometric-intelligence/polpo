# TODO: create script to check registration time with no decimation

import json

import polpo.preprocessing.dict as ppdict
from polpo.dataset import Dataset, NestedDataset
from polpo.protocol.mixin import MeshPreprocessorMixin
from polpo.surface_mesh.deformetrica import FrechetMean, LddmmMetric, Point
from polpo.surface_mesh.varifold.tuning.geometry_based import SigmaFromLengths
from polpo.time import Timer

# TODO: add script to collect times
# TODO: use JsonDict?


class LddmmToGlobal(MeshPreprocessorMixin):
    def __init__(
        self,
        known_correspondences,
        results_dir,
        ratio_kernel=1.5,
        ratio_charlen_mesh=2.0,
        ratio_charlen=0.25,
        params=None,
    ):
        self.version = "0.2.0"

        self.timer = Timer()

        self.known_correspondences = known_correspondences
        self.results_dir = results_dir

        self.ratio_kernel = ratio_kernel
        self.ratio_charlen = ratio_charlen
        self.ratio_charlen_mesh = ratio_charlen_mesh

        self._params = params or {}
        self.reset()

    def reset(self):
        self.results_ = {"version": self.version}
        self.params_ = {"version": self.version}
        self.params_.update(self._params)
        self.timer.reset()

    def tune_kernel(self, nested_meshes):
        # select varifold kernel using a randomly selected mesh per subject
        self.timer.start("tuning")

        sigma_search = SigmaFromLengths(
            ratio_charlen_mesh=self.ratio_charlen_mesh,
            ratio_charlen=self.ratio_charlen,
        )

        # TODO: add random_state?
        selected_meshes = nested_meshes.sample_inner()
        sigma_search.fit(selected_meshes.flatten().values_list())

        self.timer.stop("tuning")

        sigma_var = sigma_search.sigma_
        sigma_vel = self.ratio_kernel * sigma_var

        # TODO: also add registration kwargs
        self.params_["kernel_tuning"] = {
            "ratio_kernel": self.ratio_kernel,
            "ratio_charlen_mesh": self.ratio_charlen_mesh,
            "ratio_charlen": self.ratio_charlen,
        }
        self.results_["kernel_tuning"] = {
            "sigma_vel": sigma_vel,
            "sigma_var": sigma_var,
            "meshes": selected_meshes.flatten().keys_list(),
        }

        return sigma_vel, sigma_var

    def instantiate_metric(self, sigma_vel, sigma_var):
        registration_kwargs = dict(
            kernel_width=sigma_vel,
            regularisation=1.0,
            max_iter=2000,
            freeze_control_points=False,
            metric="varifold",
            tol=1e-16,
            attachment_kernel_width=sigma_var,
        )

        metric = LddmmMetric(self.results_dir, **registration_kwargs)

        self.params_["dirs"] = metric.dir_config.to_dict()

        return metric

    def meshes_as_points(self, nested_meshes, metric):
        return nested_meshes.map_items(
            lambda outer_key, inner_key, mesh: Point(
                id_=f"{outer_key}-{inner_key}",
                pv_surface=mesh,
                dirname=metric.dir_config.meshes_dir,
            )
        )

    def build_local_atlases(self, nested_points, metric, atlas_keys):
        # TODO: parallelize?
        estimator = FrechetMean(
            metric,
            initial_step_size=1e-1,  # TODO: pass this? at least store in params
        )

        self.timer.start("local_atlases")

        atlases = {}
        for outer_key, points in nested_points.items():
            filt_keys = atlas_keys[outer_key]
            filt_points = ppdict.SelectKeySubset(filt_keys)(points)

            filt_points = list(filt_points.values())
            estimator.fit(filt_points, atlas_id=outer_key)
            atlas = estimator.estimate_

            atlases[outer_key] = atlas

        self.timer.stop("local_atlases")

        return Dataset(atlases)

    def build_global_atlas(self, local_atlases, metric):
        estimator = FrechetMean(
            metric,
            initial_step_size=1e-1,  # TODO: pass this? at least store in params
        )

        with self.timer("global_atlas"):
            estimator.fit(local_atlases.values_list(), "gl")

        return estimator.estimate_

    def register_and_transport(self, nested_points, metric, atlas, local_atlases):
        # TODO: parallelize
        self.timer.start("register_and_transport")

        global_reprs = {}
        point_a = atlas
        for outer_key, points in nested_points.items():
            global_reprs[outer_key] = reprs = {}
            point_b = local_atlases[outer_key]

            vec_ba = metric.log(point_a, point_b)

            for inner_key, point_c in points.items():
                vec_bc = metric.log(point_c, point_b)

                trans_vec_bc = metric.parallel_transport(
                    vec_bc, point_b, direction=vec_ba
                )

                reprs[inner_key] = metric.exp(trans_vec_bc, point_a)

        self.timer.stop("register_and_transport")

        return NestedDataset(global_reprs)

    def write(self):
        with open(self.results_dir / "params.json", "w") as file:
            json.dump(self.params_, file, indent=2)

        with open(self.results_dir / "results.json", "w") as file:
            json.dump(self.results_, file, indent=2)

        with open(self.results_dir / "time.json", "w") as file:
            json.dump(self.timer.as_dict(), file, indent=2)

    def run(self, nested_meshes, atlas_keys):
        # nested_meshes: dict or polpo.dataset.NestedDataset
        if isinstance(nested_meshes, dict):
            nested_meshes = NestedDataset(nested_meshes)

        self.reset()

        self.timer.start("run")

        nested_meshes_ = self.preprocess_meshes(nested_meshes.flatten()).nest()

        sigma_vel, sigma_var = self.tune_kernel(nested_meshes_)
        metric = self.instantiate_metric(sigma_vel, sigma_var)

        nested_points = self.meshes_as_points(nested_meshes_, metric)

        local_atlases = self.build_local_atlases(nested_points, metric, atlas_keys)
        atlas = self.build_global_atlas(local_atlases, metric)

        _ = self.register_and_transport(
            nested_points,
            metric,
            atlas,
            local_atlases,
        )

        self.timer.stop("run")

        self.write()

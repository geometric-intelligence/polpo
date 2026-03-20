# TODO: create script to check registration time with no decimation

import json
import string

import polpo.preprocessing.dict as ppdict
import polpo.utils as putils
from polpo.mesh.deformetrica import FrechetMean, LddmmMetric, Point
from polpo.mesh.surface import PvSurface
from polpo.mesh.varifold.tuning.geometry_based import SigmaFromLengths
from polpo.preprocessing.mesh.registration import RigidAlignment
from polpo.time import Timer


class LddmmToGlobal:
    def __init__(
        self,
        known_correspondences,
        results_dir,
        ratio_kernel=1.5,
        ratio_charlen_mesh=2.0,
        ratio_charlen=0.25,
    ):
        self.timer = Timer()

        self.known_correspondences = known_correspondences
        self.results_dir = results_dir

        self.ratio_kernel = ratio_kernel
        self.ratio_charlen = ratio_charlen
        self.ratio_charlen_mesh = ratio_charlen_mesh

        self.reset()

    def reset(self):
        self.results_ = {"version": "0.1.0"}
        self.params_ = {}

    def map_keys(self, nested_dataset):
        # makes naming manageable
        outer_key_map = {
            outer_key: string.ascii_uppercase[index]
            for index, outer_key in enumerate(nested_dataset.keys())
        }

        inner_key_maps = {}
        for outer_key, inner_dict in nested_dataset.items():
            inner_key_maps[outer_key] = {
                key: index for index, key in enumerate(inner_dict.keys())
            }

        self.results_["key_map"] = {
            "outer": outer_key_map,
            "inner": inner_key_maps,
        }

        mapped_nested_dataset = putils.rekey_nested_dict(
            nested_dataset, outer_key_map, inner_key_maps
        )
        return (
            mapped_nested_dataset,
            outer_key_map,
            inner_key_maps,
        )

    def map_atlases_keys(self, atlases_keys, outer_key_map, inner_key_maps):
        mapped_atlases_keys = {}

        for outer_key, atlas_keys in atlases_keys.items():
            inner_map = inner_key_maps[outer_key]
            mapped_atlases_keys[outer_key_map[outer_key]] = [
                inner_map[inner_key] for inner_key in atlas_keys
            ]

        self.results_["atlases_keys"] = {
            "original": atlases_keys,
            "mapped": mapped_atlases_keys,
        }

        return mapped_atlases_keys

    def preprocess_meshes(self, nested_meshes):
        # rigidly aligns all the meshes against a randomly chosen target

        # TODO: add decimation?

        self.timer.start("prep")

        outer_key = putils.extract_random_key(nested_meshes)
        inner_key = putils.extract_random_key(nested_meshes[outer_key])

        align_pipe = RigidAlignment(
            target=nested_meshes[outer_key][inner_key],
            known_correspondences=self.known_correspondences,
        )

        nested_meshes_ = (ppdict.DictMap(align_pipe + ppdict.DictMap(PvSurface)))(
            nested_meshes
        )

        self.timer.stop("prep")

        self.results_["rigid_alignment"] = {
            "outer_key": outer_key,
            "inner_key": inner_key,
        }
        self.params_["rigid_alignment"] = {
            "known_correspondences": self.known_correspondences,
        }

        return nested_meshes_

    def tune_kernel(self, nested_meshes):
        # select varifold kernel using a randomly selected mesh per subject
        self.timer.start("tuning")

        sigma_search = SigmaFromLengths(
            ratio_charlen_mesh=self.ratio_charlen_mesh,
            ratio_charlen=self.ratio_charlen,
        )

        mesh_per_outer = []
        keys = []
        for outer_key, meshes in nested_meshes.items():
            inner_key = putils.extract_random_key(meshes)
            mesh_per_outer.append(meshes[inner_key])

            keys.append(f"{outer_key}-{inner_key}")

        sigma_search.fit(mesh_per_outer)

        self.timer.stop("tuning")

        sigma_var = sigma_search.sigma_
        sigma_vel = self.ratio_kernel * sigma_var

        self.results_["kernel_tuning"] = {
            "sigma_vel": sigma_vel,
            "sigma_var": sigma_var,
            "meshes": keys,
        }
        self.params_["kernel_tuning"] = {
            "ratio_kernel": self.ratio_kernel,
            "ratio_charlen_mesh": self.ratio_charlen_mesh,
            "ratio_charlen": self.ratio_charlen,
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

        return LddmmMetric(self.results_dir, **registration_kwargs)

    def meshes_as_points(self, metric, nested_meshes):
        nested_points = {}
        for outer_key, meshes in nested_meshes.items():
            points = nested_points[outer_key] = {}
            for inner_key, mesh in meshes.items():
                points[inner_key] = Point(
                    id_=f"{outer_key}-{inner_key}",
                    pv_surface=mesh,
                    dirname=metric.meshes_dir,
                )

        return nested_points

    def build_local_atlases(self, metric, nested_points, atlases_keys):
        estimator = FrechetMean(
            metric,
            initial_step_size=1e-1,  # TODO: pass this? at least store in params
        )

        self.timer.start("local_atlases")

        atlases = {}
        for outer_key, points in nested_points.items():
            filt_keys = atlases_keys[outer_key]
            filt_points = ppdict.SelectKeySubset(filt_keys)(points)

            filt_points = list(filt_points.values())
            if len(filt_points) == 1:
                atlas = Point(
                    id_=outer_key,
                    pv_surface=filt_points[0].as_pv(),
                    dirname=metric.atlas_dir,
                )
            else:
                estimator.fit(filt_points, atlas_id=outer_key)
                atlas = estimator.estimate_

            atlases[outer_key] = atlas

        self.timer.stop("local_atlases")

        return atlases

    def build_global_atlas(self, metric, local_atlases):
        estimator = FrechetMean(
            metric,
            initial_step_size=1e-1,  # TODO: pass this? at least store in params
        )

        with self.timer("global_atlas"):
            estimator.fit(list(local_atlases.values()), "gl")

        return estimator.estimate_

    def register_and_transport(self, metric, nested_points, atlas, local_atlases):
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

        return global_reprs

    def write(self):
        with open(self.results_dir / "params.json", "w") as file:
            json.dump(self.params_, file, indent=4)

        with open(self.results_dir / "results.json", "w") as file:
            json.dump(self.results_, file, indent=4)

        with open(self.results_dir / "time.json", "w") as file:
            json.dump(self.timer.as_dict(), file, indent=4)

    def run(self, nested_meshes, atlases_keys):
        # dataset: subject, session

        self.reset()

        self.timer.start("run")

        nested_meshes, outer_key_map, inner_key_maps = self.map_keys(nested_meshes)
        mapped_atlases_keys = self.map_atlases_keys(
            atlases_keys, outer_key_map, inner_key_maps
        )
        nested_meshes_ = self.preprocess_meshes(nested_meshes)

        sigma_vel, sigma_var = self.tune_kernel(nested_meshes_)
        metric = self.instantiate_metric(sigma_vel, sigma_var)

        nested_points = self.meshes_as_points(metric, nested_meshes_)

        local_atlases = self.build_local_atlases(
            metric, nested_points, mapped_atlases_keys
        )
        atlas = self.build_global_atlas(metric, local_atlases)

        _ = self.register_and_transport(
            metric,
            nested_points,
            atlas,
            local_atlases,
        )

        self.timer.stop("run")

        self.write()

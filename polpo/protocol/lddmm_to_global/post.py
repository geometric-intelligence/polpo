import polpo.utils as putils
from polpo.mesh.deformetrica.repr import (
    DummyPoint,
    DummyVec,
    Point,
    ShootedPoint,
    TangentVecFromRegistration,
)
from polpo.mesh.surface import PvSurface


def get_key_maps(results_data):
    outer_key_map = results_data["key_map"]["outer"]
    inner_key_maps = results_data["key_map"]["inner"]

    inv_outer_key_map = putils.invert_dict(outer_key_map)

    inv_inner_key_maps = {}
    for m_key, key in inv_outer_key_map.items():
        inv_inner_key_maps[m_key] = putils.invert_dict(inner_key_maps[key])

    return outer_key_map, inner_key_maps, inv_outer_key_map, inv_inner_key_maps


def collect_meshes(
    outer_keys,
    inner_key_maps,
    meshes_dir,
    as_pv=True,
):
    meshes = {}
    for outer_key in outer_keys:
        for inner_key in inner_key_maps[outer_key]:
            point_id = f"{outer_key}-{inner_key}"
            point = Point(id_=point_id, dirname=meshes_dir)

            if as_pv:
                point = PvSurface(point.as_pv())

            meshes[point_id] = point

    return meshes


def collect_rec_meshes_local(
    outer_keys,
    inner_key_maps,
    registration_dir,
    as_pv=True,
):
    meshes = {}
    for outer_key in outer_keys:
        for inner_key in inner_key_maps[outer_key]:
            point_id = f"{outer_key}-{inner_key}"

            out = TangentVecFromRegistration(
                base_point=DummyPoint(outer_key),
                point=DummyPoint(point_id),
                outputs_dir=registration_dir,
            )

            if as_pv:
                out = PvSurface(out.reconstructed().as_pv())

            meshes[point_id] = out

    return meshes


def collect_rec_meshes_global(
    outer_keys,
    inner_key_maps,
    shoot_dir,
    as_pv=True,
):
    meshes = {}

    for outer_key in outer_keys:
        for inner_key in inner_key_maps[outer_key]:
            point_id = f"{outer_key}-{inner_key}"

            # TODO: assuming fan
            vec_id = (
                f"{outer_key}_to_{outer_key}-{inner_key}_along_fan_{outer_key}_to_gl"
            )

            shooted_point = ShootedPoint(
                base_point=DummyPoint("gl"),
                tangent_vec=DummyVec(vec_id),
                outputs_dir=shoot_dir,
            )

            if as_pv:
                shooted_point = PvSurface(shooted_point.as_pv())

            meshes[point_id] = shooted_point

    return meshes

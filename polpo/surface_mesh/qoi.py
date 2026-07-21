import geomstats.backend as gs


def bounding_sphere_radius(surfaces):
    # TODO: handle gs style

    # list[Trimesh]
    dists = []
    for surface in surfaces:
        bounds = surface.bounding_sphere.bounds
        dists.append((bounds[1] - bounds[0])[0] / 2)

    return dists


def centroid2farthest_vertex(surfaces):
    # TODO: handle gs style
    return gs.array(
        [
            gs.amax(gs.linalg.norm(surface.vertex_centroid - surface.vertices, axis=-1))
            for surface in surfaces
        ]
    )


def vertexwise_euclidean(surface_a, surface_b):
    return gs.linalg.norm(surface_a.vertices - surface_b.vertices, axis=-1)

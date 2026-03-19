import numpy as np
import pyvista as pv


def create_blob(
    resolution=20, n_bumps=5, bump_amp=0.18, bump_width=0.7, smoothing_iter=30
):
    r"""Generate blob.

    .. math::

        r(u)=r_0+\sum_{j=1}^k a_j \exp \left(-\frac{d\left(u, c_j\right)^2}{2 \sigma_j^2}\right)

    Notes
    -----
    * good amplitude range 0.05 to 0.25
        * 0.02-0.08: barely perturbed sphere
        * 0.08-0.18: smooth blob
        * 0.2-0.25: strong lobes
    * good width range 0.4, 0.9
    """
    mesh = pv.Sphere(
        theta_resolution=resolution,
        phi_resolution=resolution,
        radius=1.0,
    )
    mesh = mesh.triangulate()

    pts = mesh.points
    U = pts / np.linalg.norm(pts, axis=1, keepdims=True)

    # smooth bump field on the sphere
    centers = np.random.randn(n_bumps, 3)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    amps = bump_amp * np.random.randn(n_bumps)

    disp = np.zeros(len(U))
    for c, a in zip(centers, amps):
        angle = np.arccos(np.clip(U @ c, -1.0, 1.0))
        disp += a * np.exp(-(angle**2) / (2 * bump_width**2))

    mesh["disp"] = disp
    blob = mesh.compute_normals(auto_orient_normals=True).warp_by_scalar("disp")

    if smoothing_iter:
        blob = blob.smooth_taubin(n_iter=smoothing_iter)

    return blob

import numpy as np
import pyvista as pv


def create_sphere(resolution=20, radius=1.0):
    """Create a triangulated sphere.

    Parameters
    ----------
    resolution : int
        Number of points in the azimuthal and polar directions.
    radius : float
        Sphere radius.

    Returns
    -------
    mesh : pyvista.PolyData
        Triangulated sphere.
    """
    mesh = pv.Sphere(
        theta_resolution=resolution,
        phi_resolution=resolution,
        radius=radius,
    )
    return mesh.triangulate()


def create_icosphere(nsub=3, radius=1.0):
    """Create a triangulated sphere by subdividing an icosahedron.

    Parameters
    ----------
    nsub : int
        Number of recursive subdivisions of the initial icosahedron.
    radius : float
        Sphere radius.

    Returns
    -------
    mesh : pyvista.PolyData
        Triangulated icosphere.
    """
    return pv.Icosphere(
        radius=radius,
        nsub=nsub,
    )


def create_bump_field(
    n_bumps=5,
    bump_amp=0.18,
    bump_width=0.7,
    random_state=None,
):
    """Create a random bump field.

    Parameters
    ----------
    n_bumps : int
        Number of bumps.
    bump_amp : float
        Scale of the random bump amplitudes.
    bump_width : float
        Width of the bumps in radians.
    random_state : int or numpy.random.Generator
        Random state used to generate the bump field.

    Returns
    -------
    field : BumpField
        Random bump field.

    Notes
    -----
    Typical values for ``bump_amp`` are:

    * 0.02--0.08: mild perturbations.
    * 0.08--0.18: moderate deformations.
    * 0.20--0.25: pronounced lobes.

    Typical values for ``bump_width`` are 0.4--0.9 radians.
    """
    rng = np.random.default_rng(random_state)

    centers = rng.standard_normal((n_bumps, 3))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    amplitudes = bump_amp * rng.standard_normal(n_bumps)

    return BumpField(
        centers=centers,
        amplitudes=amplitudes,
        width=bump_width,
    )


class BumpField:
    r"""Smooth radial bump field on the sphere.

    The field is defined as a sum of Gaussian kernels,

    .. math::

        b(u)
        =
        \sum_{j=1}^k
        a_j
        \exp\left(
            -\frac{
                d_{\mathbb{S}^2}(u, c_j)^2
            }{
                2 \sigma^2
            }
        \right),

    where :math:`u \in \mathbb{S}^2`, :math:`c_j` are the bump centers,
    :math:`a_j` are the bump amplitudes, :math:`\sigma` is the bump width,
    and

    .. math::

        d_{\mathbb{S}^2}(u, c)
        =
        \arccos(u^\top c)

    is the geodesic distance on the unit sphere.

    Parameters
    ----------
    centers : array-like, shape=(n_bumps, 3)
        Unit vectors defining the bump centers.
    amplitudes : array-like, shape=(n_bumps,)
        Bump amplitudes.
    width : float
        Width of the Gaussian kernels in radians.
    """

    def __init__(self, centers, amplitudes, width):
        self.centers = centers
        self.amplitudes = amplitudes
        self.width = width

    def evaluate(self, directions):
        """Evaluate the bump field at unit directions.

        Parameters
        ----------
        directions : array-like, shape=(n_points, 3)
            Unit directions on the sphere.

        Returns
        -------
        values : array-like, shape=(n_points,)
            Radial displacement at each direction.
        """
        values = np.zeros(len(directions))

        for center, amplitude in zip(self.centers, self.amplitudes):
            angle = np.arccos(np.clip(directions @ center, -1.0, 1.0))
            values += amplitude * np.exp(-(angle**2) / (2 * self.width**2))

        return values

    def apply(self, mesh):
        """Apply the bump field to a spherical mesh.

        Parameters
        ----------
        mesh : pyvista.PolyData
            Spherical mesh to deform.

        Returns
        -------
        blob : pyvista.PolyData
            Mesh displaced according to the bump field.
        """
        mesh = mesh.copy()

        directions = mesh.points / np.linalg.norm(mesh.points, axis=1, keepdims=True)
        mesh["disp"] = self.evaluate(directions)

        return mesh.compute_normals(auto_orient_normals=True).warp_by_scalar("disp")


def create_blob(
    resolution=20,
    n_bumps=5,
    bump_amp=0.18,
    bump_width=0.7,
    smoothing_iter=30,
):
    """Create a smooth random blob.

    Parameters
    ----------
    resolution : int
        Resolution of the initial sphere.
    n_bumps : int
        Number of bumps.
    bump_amp : float
        Scale of the bump amplitudes.
    bump_width : float
        Width of the bumps in radians.
    smoothing_iter : int
        Number of Taubin smoothing iterations.

    Returns
    -------
    blob : pyvista.PolyData
        Generated blob.
    """
    sphere = create_sphere(resolution=resolution)

    bump_field = create_bump_field(
        n_bumps=n_bumps,
        bump_amp=bump_amp,
        bump_width=bump_width,
    )
    blob = bump_field.apply(sphere)

    if smoothing_iter:
        blob = blob.smooth_taubin(n_iter=smoothing_iter)

    return blob

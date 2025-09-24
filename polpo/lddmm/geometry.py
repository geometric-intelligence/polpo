import logging

import support.kernels as kernel_factory
from core.model_tools.deformations.exponential import Exponential
from launch.compute_parallel_transport import (
    compute_parallel_transport,
    compute_pole_ladder,
)
from launch.compute_shooting import compute_shooting
from support import utilities

import polpo.lddmm.registration as registration
import polpo.lddmm.strings as strings

logger = logging.getLogger(__name__)


def _warn_usunused_kwargs(func_name, kwargs, unused):
    unused_used = [arg for arg in unused if arg in kwargs]
    if not unused_used:
        return

    msg = f"The following args are ignored by {func_name}: {', '.join(unused_used)}"
    logger.warn(msg)


def pole_ladder(
    control_points,
    momenta,
    momenta_to_transport,
    output_dir,
    kernel_width=15,
    kernel_type="torch",
    control_points_to_transport=None,
    concentration_of_time_points=10,
    tmin=0,
    tmax=1,
    **model_parameters,
):
    """Compute parallel transport of a tangent vector along a geodesic with the pole ladder.

    Transports a tangent vector along a geodesic (called main geodesic). Both must have been
    estimated by using the `registration` function. The main geodesic must be estimated using RK4
    steps. Kernel parameters should match the ones used in the registration function.

    This function performs the actual parallel transport computation using pre-computed control points and momenta.
    It takes as input the control points and momenta that define both:
    1. The main geodesic along which to transport
    2. The tangent vector to be transported (which also corresponds to a geodesic).

    Related
    -------
    The estimate_parallel_transport() function provides a higher-level interface that handles the full parallel transport
    pipeline including the registration steps needed to obtain the control points and momenta. It computes geodesics
    between three shapes (atlas, source, target) and uses this transport() function as the final step.

    Parameters
    ----------
    control_points: str or pathlib.Path
        Path to the txt file that contains the initial control points for the main geodesic.
    momenta: str or pathlib.Path
        Path to the txt file that contains the initial momenta for the main geodesic.
    control_points_to_transport: str or pathlib.Path
        Path to the txt file that contains the initial control points of the deformation to
        transport.
    momenta_to_transport: str or pathlib.Path
        Path to the txt file that contains the initial momenta to transport.
    output_dir: str or pathlib.Path
        Path to a directory where results will be saved. It will be created if it does not
        already exist.
    kernel_width: float
        Width of the Gaussian kernel. Controls the spatial smoothness of the deformation and
        influences the number of parameters required to represent the deformation.
        Optional, default: 20.
    kernel_type: str, {torch, keops}
        Package to use for convolutions of velocity fields and loss functions.
    """
    # TODO: do wrapper function? outputs are very different
    # NB: returns only the final transported quantities

    output_dir.mkdir(parents=True, exist_ok=True)

    transported_cp, transported_mom = compute_pole_ladder(
        initial_control_points=control_points,
        initial_momenta=momenta,
        initial_momenta_to_transport=momenta_to_transport,
        initial_control_points_to_transport=control_points_to_transport,
        number_of_time_points=concentration_of_time_points + 1,
        deformation_kernel_type=kernel_type,
        deformation_kernel_width=kernel_width,
        output_dir=output_dir,
        tmin=tmin,
        tmax=tmax,
        **model_parameters,
    )
    return transported_cp, transported_mom


def shoot(
    source,
    control_points,
    momenta,
    output_dir,
    kernel_width=20.0,
    kernel_type="torch",
    kernel_device="cuda",
    **model_options,
):
    """Compute geodesic.

    Compute the deformation of a source shape by the flow parametrized by control points and
    momenta.

    Parameters
    ----------
    source: str or pathlib.Path
        Path to the vtk file that contains the source mesh.
    control_points: str or pathlib.Path
        Path to the txt file that contains the initial control points.
    momenta: str or pathlib.Path
        Path to the txt file that contains the initial momenta.
    kernel_width: float
        Width of the Gaussian kernel. Controls the spatial smoothness of the deformation and
        influences the number of parameters required to represent the deformation.
        Optional, default: 20.
    kernel_type: str, {torch, keops}
        Package to use for convolutions of velocity fields and loss functions.
    kernel_device: str, {cuda, cpu}
    """
    # cp, momenta, meshes (cp and momenta writing controlled by write_adjoint_parameters)
    # allows for multiple momenta
    # filenames as int, i.e. connection to ids is lost

    _warn_usunused_kwargs(
        "shoot",
        model_options,
        unused=(
            "tensor_integer_type",
            "deformation_kernel_device",
            "number_of_time_points",
        ),
    )

    template_specifications = {
        "shape": {
            "deformable_object_type": "SurfaceMesh",
            "noise_std": -1,  # not used
            "filename": source,
        }
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    compute_shooting(
        template_specifications,
        initial_control_points=control_points,
        initial_momenta=momenta,
        deformation_kernel_width=kernel_width,
        deformation_kernel_type=kernel_type,
        output_dir=output_dir,
        **model_options,
    )


def parallel_transport(
    source,
    control_points,
    momenta,
    momenta_to_transport,
    output_dir,
    kernel_width=1.0,
    control_points_to_transport=None,
    kernel_type="torch",
    kernel_device="cuda",
    **model_options,
):
    _warn_usunused_kwargs(
        "parallel_transport",
        model_options,
        unused=("tensor_integer_type",),
    )

    template_specifications = {
        "shape": {
            "deformable_object_type": "SurfaceMesh",
            "noise_std": -1,  # not used
            "filename": source,
        }
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    return compute_parallel_transport(
        template_specifications,
        output_dir=output_dir,
        deformation_kernel_width=kernel_width,
        initial_control_points=control_points,
        initial_momenta=momenta,
        initial_momenta_to_transport=momenta_to_transport,
        initial_control_points_to_transport=control_points_to_transport,
        deformation_kernel_type=kernel_type,
        **model_options,
    )


def parallel_transport_from_meshes(
    source,
    target,
    atlas,
    name,
    output_dir,
    registration_args,
    transport_args,
    main_reg_dir=None,
):
    """
    Estimate the parallel transport of the "time deformation" along the "subject-to-patient" deformation.

    This function computes parallel transport using as inputs three points on the manifold:
    1. Computes a geodesic from the atlas to the source shape (the "main geodesic")
    2. Computes a geodesic from the source to the target shape (the "time deformation")
    3. Parallel transports the tangent vector of the time deformation along the main geodesic

    Related
    -------
    The transport() function performs the actual parallel transport computation using only control points and momenta.
    While estimate_parallel_transport() function estimates the full parallel transport pipeline including registrations,
    transport() is the lower-level function that takes pre-computed control points and momenta as inputs to execute just the transport step.


    Parameters
    ----------
    source: str or pathlib.Path
        Path to the vtk file that contains the source mesh for the geodesic that will be transported. Inintial time step.
    target: str or pathlib.Path
        Path to the vtk file that contains the target mesh of the geodesic that will be transported. Final time step.
    atlas: str or pathlib.Path
        Path to the vtk file that contains the atlas mesh. The geodesic of atlas towards source is geodesic along which the time deformation will be transported.
    name: str
        Name of the deformation.
    output_dir: str or pathlib.Path
        Path to the directory where the results will be saved.
    registration_args: dict
        Arguments for the registration function.
    transport_args: dict
        Arguments for the transport function.
    main_reg_dir: str or pathlib.Path
        Path to the directory where the results of the main registration will be saved.
        The main registration is the subject-to-patient deformation that is the geodesic along which the time deformation will be transported.
        If None, the results will be saved in the output_dir directory.
    """
    # estimation of time deformation: this is the deformation that will be transported
    time_reg_dir = output_dir / name / f"time_reg_{name}"
    time_reg_dir.mkdir(parents=True, exist_ok=True)

    registration.estimate_registration(
        source, target, time_reg_dir, **registration_args
    )

    if main_reg_dir is None:
        # estimation of subject-to-patient deformation: this is the geodesic along which the time deformation will be transported
        main_reg_dir = output_dir / name / f"main_reg_{name}"
        main_reg_dir.mkdir(parents=True, exist_ok=True)

        registration.estimate_registration(
            atlas, source, main_reg_dir, **registration_args
        )

    # parallel transport of time deformation along subject-to-patient deformation
    momenta_to_transport = (time_reg_dir / strings.momenta_str).as_posix()
    control_points_to_transport = (time_reg_dir / strings.cp_str).as_posix()
    control_points = (main_reg_dir / strings.cp_str).as_posix()
    momenta = (main_reg_dir / strings.cp_str).as_posix()

    transport_dir = output_dir / name / "transport"
    transport_dir.mkdir(parents=True, exist_ok=True)
    cp, mom = pole_ladder(
        control_points,
        momenta,
        control_points_to_transport,
        momenta_to_transport,
        transport_dir,
        **transport_args,
    )

    return cp, mom


def flow(
    base_point,
    control_points_t,
    momenta_t,
    kernel_width=20.0,
    kernel_type="torch",
    use_rk2_for_flow=False,
    use_rk2_for_shoot=False,
):
    # NB: not working!
    # flow source along trajectory
    # mostly for debugging and understanding
    # just a hack

    deformation_kernel = kernel_factory.factory(
        kernel_type,
        kernel_width=kernel_width,
    )
    exponential = Exponential(
        kernel=deformation_kernel,
        use_rk2_for_shoot=use_rk2_for_shoot,
        use_rk2_for_flow=use_rk2_for_flow,
    )

    # TODO: need to control number of time points?

    dtype = "float32"
    device, _ = utilities.get_best_device()
    tensor_scalar_type = utilities.get_torch_scalar_type(dtype)

    move_data = lambda x: utilities.move_data(
        x,
        dtype=tensor_scalar_type,
        device=device,
    )

    template = {"landmark_points": move_data(base_point.points)}

    exponential.set_initial_template_points(template)

    exponential.control_points_t = move_data(control_points_t)
    exponential.momenta_t = move_data(momenta_t)
    exponential.number_of_time_points = len(momenta_t)

    exponential.shoot_is_modified = False

    exponential.flow()

    return exponential.template_points_t["landmark_points"]

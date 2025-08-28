import subprocess
import time

from api.deformetrica import Deformetrica


import polpo.lddmm.registration as registration
import polpo.lddmm.strings as strings

from launch.compute_parallel_transport import compute_pole_ladder
from launch.compute_shooting import compute_shooting


def transport(
    control_points,
    momenta,
    control_points_to_transport,
    momenta_to_transport,
    output_dir,
    kernel_type="torch",
    kernel_width=15,
    kernel_device="cuda",
    n_rungs=10,
):
    """Compute parallel transport with the pole ladder.

    Transports a tangent vector along a geodesic (called main geodesic). Both must have been
    estimated by using the `registration` function. The main geodesic must be estimated using RK4
    steps. Kernel parameters should match the ones used in the registration function.

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
    kernel_device: str, {cuda, cpu}
    n_rungs: int
        Number of discretization steps in the pole ladder algorithm. Should match
        number_of_time_points in the registration of the main geodesic.
        Optional, default: 10.
    """
    Deformetrica(output_dir, verbosity="INFO")

    deformation_parameters = {
        "deformation_kernel_type": kernel_type,
        "deformation_kernel_width": kernel_width,
        "deformation_kernel_device": kernel_device,
        "concentration_of_time_points": n_rungs,
        "number_of_time_points": n_rungs + 1,
        "tmin": 0,
        "tmax": 1,
        "output_dir": output_dir,
    }

    transported_cp, transported_mom = compute_pole_ladder(
        initial_control_points=control_points,
        initial_momenta=momenta,
        initial_momenta_to_transport=momenta_to_transport,
        initial_control_points_to_transport=control_points_to_transport,
        **deformation_parameters,
    )
    return transported_cp, transported_mom


def shoot(
    source,
    control_points,
    momenta,
    output_dir,
    kernel_width=20.0,
    regularisation=1.0,
    number_of_time_steps=10,
    kernel_type="torch",
    kernel_device="cuda",
    write_params=True,
    deformation="geodesic",
    external_forces=None,
    use_rk2_for_flow=False,
    use_rk2_for_shoot=False,
):
    """Exponential map.

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
    regularisation: unused.
    write_params: bool
    external_forces: str or pathlib.Path
        Path to the vtk file that contains the external forces to compute a spline deformation.
    use_rk2_for_flow: bool
        Whether to use Runge-Kutta order 2 steps in the integration of the flow equation, i.e. when
        warping the shape. If False, a Euler step is used.
        Optional, default: False
    use_rk2_for_shoot: bool
        Whether to use Runge-Kutta order 2 steps in the integration of the Hamiltonian equation that
        governs the time evolution of control points and momenta. If False, a Euler step is used.
        Optional, default: False
    """
    deformation_parameters = {
        "deformation_model": deformation,
        "deformation_kernel_type": kernel_type,
        "deformation_kernel_width": kernel_width,
        "deformation_kernel_device": kernel_device,
        "concentration_of_time_points": number_of_time_steps,
        "number_of_time_points": number_of_time_steps + 1,
        "use_rk2_for_flow": use_rk2_for_flow,
        "use_rk2_for_shoot": use_rk2_for_shoot,
        "output_dir": output_dir,
        "write_adjoint_parameters": write_params,
    }

    template_specifications = {
        "shape": {
            "deformable_object_type": "landmark",
            "kernel_type": kernel_type,
            "kernel_width": kernel_width,
            "kernel_device": kernel_device,
            "noise_std": regularisation,
            "filename": source,
            "noise_variance_prior_scale_std": None,
            "noise_variance_prior_normalized_dof": 0.01,
        }
    }

    Deformetrica(output_dir, verbosity="INFO")
    compute_shooting(
        template_specifications,
        initial_control_points=control_points,
        external_forces=external_forces,
        initial_momenta=momenta,
        **deformation_parameters,
    )

    return time.gmtime()

def estimate_parallel_transport(
        source, 
        target, 
        atlas, 
        name, 
        output_dir, 
        registration_args,
        transport_args, 
        shoot_args, 
        main_reg_dir=None):
    # estimation of time deformation
    time_reg_dir = output_dir / name / f'time_reg_{name}'
    time_reg_dir.mkdir(parents=True, exist_ok=True)

    registration.estimate_registration(
        source, target, time_reg_dir, **registration_args)

    if main_reg_dir is None:
        # estimation of subject-to-patient deformation
        main_reg_dir = output_dir / name / f'main_reg_{name}'
        main_reg_dir.mkdir(parents=True, exist_ok=True)

        registration.estimate_registration(
            atlas, source, main_reg_dir, **registration_args)

    # parallel transport of time deformation along subject-to-patient deformation
    momenta_to_transport = (time_reg_dir / strings.momenta_str).as_posix()
    control_points_to_transport = (time_reg_dir / strings.cp_str).as_posix()
    control_points = (main_reg_dir / strings.cp_str).as_posix()
    momenta = (main_reg_dir / strings.cp_str).as_posix()

    transport_dir = output_dir / name / 'transport'
    transport_dir.mkdir(parents=True, exist_ok=True)
    cp, mom = transport(
        control_points, momenta, control_points_to_transport, momenta_to_transport,
        transport_dir, **transport_args)

    # Shoot transported momenta from atlas
    transported_cp = transport_dir / 'final_cp.txt'
    transported_mom = transport_dir / 'transported_momenta.txt'
    shoot(
        control_points=transported_cp.as_posix(),
        momenta=transported_mom.as_posix(),
        output_dir=transport_dir, **shoot_args)

    shoot_name = transport_dir / strings.shoot_str.format(transport_args['n_rungs'])
    subprocess.call(['cp', shoot_name, transport_dir / f'transported_shoot_{name}.vtk'])
    return cp, mom
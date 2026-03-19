from in_out.array_readers_and_writers import read_3D_array

import polpo.preprocessing.dict as ppdict
from polpo.preprocessing import FilteredGroupBy, Map, Sorter
from polpo.preprocessing.load.deformetrica import (
    LoadControlPointsFlow,
    LoadMomentaFlow,
    _file2mesh,
    load_vtk_mesh,  # noqa: F401
)
from polpo.preprocessing.path import FileFinder, IsFileType, PathShortener
from polpo.preprocessing.str import (
    Contains,
    ContainsAll,
    DigitFinder,
    RegexGroupFinder,
    TryToInt,
)
from polpo.utils import custom_order

# TODO: remove strings?
# TODO: functions need to be renamed

# TODO: deeply simplify this file


def build_registration_name(source, target):
    # TODO: remove
    return f"{source}->{target}"


def build_parallel_transport_name(source, geod_target, transp_target):
    # TODO: remove
    return f"{source}>{transp_target}--{source}>{geod_target}->{geod_target}"


def build_parallel_shoot_name(source, geod_target, transp_target):
    # TODO: remove
    return f"{geod_target}(t{source}>{transp_target})"


def get_deterministic_atlas_reconstruction_names(path, subset=None):
    # TODO: move

    # in DeterministicAtlas._write_model_predictions
    # name = self.name + '__Reconstruction__' + object_name + '__subject_' + subject_id + object_extension

    file_finder = FileFinder(
        rules=[IsFileType("vtk"), Contains("__Reconstruction__")],
        as_list=True,
    )

    path_to_id = RegexGroupFinder(r"__subject_([A-Za-z0-9]+)") + TryToInt()

    sorter = None
    if subset is not None:
        sorter = custom_order(subset)

    file_finder += ppdict.HashWithIncoming(
        key_step=Map(path_to_id), key_sorter=sorter, key_subset=subset
    )

    # dict[str or int: str]
    return file_finder(path)


def get_deterministic_atlas_flow_names(path, subset=None):
    # TODO: move

    # in DeterministicAtlas._write_model_predictions
    # name = self.name + '__flow__' + object_name + '__subject_' + subject_id + "__tp_" + str(j) + object_extension

    file_finder = FileFinder(
        rules=[IsFileType("vtk"), Contains("__flow__")],
    )

    path_to_id = RegexGroupFinder(r"__subject_([A-Za-z0-9]+)") + TryToInt()
    path_to_tp = PathShortener() + DigitFinder(index=-1)

    # TODO: try to call path_to_id only once if no subset
    sorter = Sorter(path_to_id) if subset is None else None

    file_finder += (
        sorter
        + FilteredGroupBy(path_to_id, subset=subset)
        + ppdict.DictMap(Sorter(path_to_tp))
    )

    # dict[str or int: list[str]]
    return file_finder(path)


def get_shooting_flow_names(path):
    # TODO: homogenize with get_deterministic_atlas_flow_names?

    file_finder = FileFinder(
        rules=[IsFileType("vtk"), Contains("__GeodesicFlow__")],
    )

    path_to_tp = (
        PathShortener() + RegexGroupFinder(pattern=r"__tp_(\d+)_") + (lambda x: int(x))
    )
    file_finder += Sorter(path_to_tp)

    # list[str]
    return file_finder(path)


def get_parallel_shooting_flow_names(path):
    file_finder = FileFinder(
        rules=[IsFileType("vtk"), Contains("parallel_curve")],
    )

    path_to_tp = (
        PathShortener() + RegexGroupFinder(pattern=r"_tp_(\d+)_") + (lambda x: int(x))
    )
    file_finder += Sorter(path_to_tp)

    # list[str]
    return file_finder(path)


def load_template(path, as_path=False, as_pv=False):
    # as_pv is ignored if as_path is True

    file_finder = FileFinder(
        rules=[IsFileType("vtk"), Contains("__EstimatedParameters__Template")],
    )

    filename = file_finder(path)

    if as_path:
        return filename

    return _file2mesh(as_pv)(filename)


def load_cp(path, as_path=False):
    possible_filenames = (
        "DeterministicAtlas__EstimatedParameters__ControlPoints.txt",
        "final_cp.txt",
    )

    for name in possible_filenames:
        cp_name = path / name
        if cp_name.exists():
            break
    else:
        raise FileNotFoundError("Can't find any control points.")

    if as_path:
        return cp_name

    return read_3D_array(cp_name)


def load_transported_cp(path, as_path=False):
    # if pole ladder
    cp_name = path / "final_cp.txt"
    if not cp_name.exists():
        cp_names = LoadControlPointsFlow(as_path=True)(path)
        cp_name = cp_names[list(cp_names.keys())[-1]]

    if as_path:
        return cp_name

    return read_3D_array(cp_name)


def get_deterministic_atlas_momenta_names(path, subset=None):
    # NB: part of the way we split it now in polpo

    file_finder = FileFinder(
        rules=[IsFileType("txt"), ContainsAll(("__Momenta__", "__subject_"))],
    )

    path_to_id = RegexGroupFinder(r"__subject_([A-Za-z0-9]+)")

    sorter = Sorter(path_to_id) if subset is None else None

    file_finder += sorter + FilteredGroupBy(path_to_id, subset=subset)

    return file_finder(path)


def load_deterministic_atlas_momenta(path, id_, as_path=False):
    filenames = get_deterministic_atlas_flow_names(path, subset=[id_])
    mom_name = ppdict.ExtractUniqueKey()(filenames)

    if as_path:
        return mom_name

    return read_3D_array(mom_name)


def load_momenta(path, as_path=False):
    # TODO: put together with load_deterministic_atlas_momenta?
    possible_filenames = [
        "DeterministicAtlas__EstimatedParameters__Momenta.txt",
        "transported_momenta.txt",
    ]

    for name in possible_filenames:
        mom_name = path / name
        if mom_name.exists():
            break
    else:
        raise FileNotFoundError("Can't find any momenta")

    if as_path:
        return mom_name

    return read_3D_array(mom_name)


def load_transported_momenta(path, as_path=False):
    # TODO: really needed?
    # if pole ladder
    mom_name = path / "transported_momenta.txt"
    if not mom_name.exists():
        mom_names = LoadMomentaFlow(as_path=True, extra_rules=Contains("Transported"))(
            path
        )
        mom_name = mom_names[list(mom_names.keys())[-1]]

    if as_path:
        return mom_name

    return read_3D_array(mom_name)


def load_deterministic_atlas_reconstructions(
    path, subset=None, as_pv=False, as_path=False
):
    # as_pv is ignored if as_path is True
    filenames = get_deterministic_atlas_reconstruction_names(path, subset=subset)
    if as_path:
        return filenames

    return ppdict.DictMap(_file2mesh(as_pv))(filenames)


def load_deterministic_atlas_reconstruction(path, as_pv=False, as_path=False, id_=None):
    # TODO: can do better

    # NB: convenient
    meshes = load_deterministic_atlas_reconstructions(
        path, as_pv=as_pv, as_path=as_path, subset=id_
    )
    return ppdict.ExtractUniqueKey()(meshes)


def load_deterministic_atlas_flows(path, subset=None, as_pv=False, as_path=False):
    # as_pv is ignored if as_path is True
    filenames = get_deterministic_atlas_flow_names(path, subset=subset)
    if as_path:
        return filenames

    return ppdict.NestedDictMap(_file2mesh(as_pv), inner_is_dict=False)(filenames)


def load_deterministic_atlas_flow(path, as_pv=False, as_path=False):
    # TODO: check use
    # TODO: improve if used
    # NB: convenient
    meshes = load_deterministic_atlas_flows(path, as_pv=as_pv, as_path=as_path)
    return ppdict.ExtractUniqueKey()(meshes)


def load_shooting_flow(path, as_pv=False, as_path=False):
    # TODO: maybe add last and reduce number methods?
    filenames = get_shooting_flow_names(path)
    if as_path:
        return filenames

    return Map(_file2mesh(as_pv))(filenames)


def load_shooted_point(path, as_pv=False, as_path=False):
    filename = get_shooting_flow_names(path)[-1]
    if as_path:
        return filename

    return _file2mesh(as_pv)(filename)


def load_parallel_shooted_point(path, as_pv=False, as_path=False):
    filename = get_parallel_shooting_flow_names(path)[-1]
    if as_path:
        return filename

    return _file2mesh(as_pv)(filename)

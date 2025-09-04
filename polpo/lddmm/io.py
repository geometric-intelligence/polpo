from in_out.array_readers_and_writers import read_3D_array
from in_out.deformable_object_reader import DeformableObjectReader

import polpo.lddmm.strings as lddmm_strings
import polpo.preprocessing.dict as ppdict
from polpo.preprocessing import (
    Filter,
    GroupBy,
    IdentityStep,
    Map,
    Sorter,
)
from polpo.preprocessing.mesh.conversion import PvFromData
from polpo.preprocessing.path import FileFinder, IsFileType
from polpo.preprocessing.str import Contains, DigitFinder, RegexGroupFinder, TryToInt
from polpo.utils import custom_order

# TODO: remove strings?


def get_template_name(path):
    return path / lddmm_strings.template_str


def get_cp_name(path):
    return path / lddmm_strings.cp_str


def get_momenta_name(path):
    return path / lddmm_strings.momenta_str


def get_deterministic_atlas_reconstruction_names(path, subset=None):
    # in DeterministicAtlas._write_model_predictions
    # name = self.name + '__Reconstruction__' + object_name + '__subject_' + subject_id + object_extension

    file_finder = FileFinder(
        rules=[IsFileType("vtk"), Contains("__Reconstruction__")],
        as_list=True,
    )

    path_to_id = RegexGroupFinder(r"__subject_([A-Za-z0-9]+)") + TryToInt()

    if subset is not None:
        file_finder += Filter(
            func=lambda folder_name: path_to_id(folder_name) in subset
        )

    if subset is None:
        sorter = Sorter(lambda x: path_to_id(x))

    else:
        _custom_order = custom_order(subset)
        sorter = Sorter(lambda x: _custom_order(path_to_id(x)))

    file_finder += sorter + ppdict.HashWithIncoming(key_step=Map(path_to_id))

    # dict[str or int: str]
    return file_finder(path)


def get_deterministic_atlas_flow_names(path, subset=None):
    # in DeterministicAtlas._write_model_predictions
    # name = self.name + '__flow__' + object_name + '__subject_' + subject_id + "__tp_" + str(j) + object_extension

    file_finder = FileFinder(
        rules=[IsFileType("vtk"), Contains("__flow__")],
    )

    path_to_id = RegexGroupFinder(r"__subject_([A-Za-z0-9]+)") + TryToInt()
    path_to_tp = DigitFinder(index=-1)

    if subset is None:
        sorter = Sorter(lambda x: (path_to_id(x), path_to_tp(x)))
        filter_ = IdentityStep()
    else:
        _custom_order = custom_order(subset)
        sorter = Sorter(lambda x: (_custom_order(path_to_id(x)), path_to_tp(x)))
        filter_ = ppdict.SelectKeySubset(subset)

    file_finder += sorter + GroupBy(path_to_id) + filter_

    # dict[str or int: list[str]]
    return file_finder(path)


def load_vtk_mesh(path):
    vertices, _, faces = DeformableObjectReader.read_vtk_file(
        path, extract_connectivity=True
    )
    return vertices, faces


def _file2mesh(as_pv=False):
    file2mesh = load_vtk_mesh
    if as_pv:
        file2mesh += PvFromData()

    return file2mesh


def load_template(path, as_pv=False):
    # just to reduce dependencies, could have used pyvista instead
    return _file2mesh(as_pv)(get_template_name(path))


def load_cp_and_momenta(path):
    cp = read_3D_array(get_cp_name(path))
    momenta = read_3D_array(get_momenta_name(path))

    return cp, momenta


def load_deterministic_atlas_reconstructions(path, subset=None, as_pv=False):
    filenames = get_deterministic_atlas_reconstruction_names(path, subset=subset)
    return ppdict.DictMap(_file2mesh(as_pv))(filenames)


def load_deterministic_atlas_reconstruction(path, as_pv=False):
    # NB: convenient
    meshes = load_deterministic_atlas_reconstructions(path, as_pv=as_pv)
    return ppdict.ExtractUniqueKey()(meshes)


def load_deterministic_atlas_flows(path, subset=None, as_pv=False):
    filenames = get_deterministic_atlas_flow_names(path, subset=subset)
    return ppdict.NestedDictMap(_file2mesh(as_pv), inner_is_dict=False)(filenames)


def load_deterministic_atlas_flow(path, as_pv=False):
    # NB: convenient
    meshes = load_deterministic_atlas_flows(path, as_pv=as_pv)
    return ppdict.ExtractUniqueKey()(meshes)

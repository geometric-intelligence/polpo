from in_out.array_readers_and_writers import read_3D_array
from in_out.deformable_object_reader import DeformableObjectReader

import polpo.lddmm.strings as lddmm_strings
import polpo.preprocessing.dict as ppdict
from polpo.preprocessing import FilteredGroupBy, IdentityStep, Map, Sorter
from polpo.preprocessing.mesh.conversion import PvFromData
from polpo.preprocessing.path import FileFinder, IsFileType
from polpo.preprocessing.str import Contains, DigitFinder, RegexGroupFinder, TryToInt
from polpo.utils import custom_order

# TODO: remove strings?
# TODO: functions need to be renamed


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
    path_to_tp = DigitFinder(index=-1)

    # TODO: try to call path_to_id only once if no subset
    sorter = Sorter(path_to_id) if subset is None else IdentityStep()

    file_finder += (
        sorter
        + FilteredGroupBy(path_to_id, subset=subset)
        + ppdict.DictMap(Sorter(path_to_tp))
    )

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


def load_template(path, as_path=False, as_pv=False):
    # as_path as precedence to as_pv
    filename = path / lddmm_strings.template_str

    if as_path:
        return filename

    return _file2mesh(as_pv)(filename)


def load_cp(path, as_path=False):
    cp_name = path / lddmm_strings.cp_str

    if as_path:
        return cp_name

    return read_3D_array(cp_name)


def load_momenta(path, as_path=False):
    mom_name = path / lddmm_strings.momenta_str

    if as_path:
        return mom_name

    return read_3D_array(mom_name)


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

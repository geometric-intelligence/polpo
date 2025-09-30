import numpy as np
from in_out.array_readers_and_writers import read_3D_array
from in_out.deformable_object_reader import DeformableObjectReader

import polpo.lddmm as plddmm
import polpo.preprocessing.dict as ppdict
import polpo.utils as putils
from polpo.preprocessing import Map
from polpo.preprocessing.mesh.conversion import PvFromData
from polpo.preprocessing.path import FileFinder, IsFileType
from polpo.preprocessing.str import Contains, RegexGroupFinder


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


def LoadMeshFlow(as_path=False, as_pv=True, extra_rules=()):
    # TODO: index by time bool

    # as_path as precedence to as_pv
    path_to_tp = RegexGroupFinder(r"_tp_(\d+)") + int

    rules = [IsFileType("vtk")] + putils.as_list(extra_rules)

    file_finder = FileFinder(
        rules=rules,
        as_list=True,
    ) + ppdict.HashWithIncoming(key_step=Map(path_to_tp), key_sorter=lambda x: x)

    if as_path:
        return file_finder

    return file_finder + ppdict.DictMap(plddmm.io._file2mesh(as_pv=as_pv))


def LoadControlPointsFlow(as_path=False, as_array=True, extra_rules=()):
    # as_path as precedence to as_array

    path_to_tp = RegexGroupFinder(r"_tp_(\d+)") + int

    rules = [IsFileType("txt"), Contains("ControlPoints")] + putils.as_list(extra_rules)

    file_finder = FileFinder(
        rules=rules,
        as_list=True,
    ) + ppdict.HashWithIncoming(key_step=Map(path_to_tp), key_sorter=lambda x: x)

    if as_path:
        return file_finder

    pipe = file_finder + ppdict.DictMap(read_3D_array)
    if not as_array:
        return pipe

    return pipe + ppdict.DictToValuesList() + np.stack


def LoadMomentaFlow(as_path=False, as_array=True, extra_rules=()):
    # as_path as precedence to as_array

    path_to_tp = RegexGroupFinder(r"_tp_(\d+)") + int

    rules = [
        IsFileType("txt"),
        Contains("Momenta"),
    ] + putils.as_list(extra_rules)

    file_finder = FileFinder(
        rules=rules,
        as_list=True,
    ) + ppdict.HashWithIncoming(key_step=Map(path_to_tp), key_sorter=lambda x: x)

    if as_path:
        return file_finder

    pipe = file_finder + ppdict.DictMap(read_3D_array)

    if not as_array:
        return pipe

    return pipe + ppdict.DictToValuesList() + np.stack

import abc
import os
import warnings

import fast_simplification
import H2_SurfaceMatch.H2_match  # noqa: E402
import H2_SurfaceMatch.utils.utils  # noqa: E402
import nibabel as nib
import numpy as np
import pandas as pd
import skimage
import trimesh
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# TODO: rename to data_processing? maybe have a folder?

# TODO: is there any difference between a DataLoader and a step?

# TODO: review naming
# TODO: replace step by pipeline?


class DataLoader(abc.ABC):
    @abc.abstractmethod
    def load(self):
        pass


class PipelineDataLoader(DataLoader):
    # TODO: accept multiple pipelines? e.g. store info
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def load(self):
        return self.pipeline.apply()


class PreprocessingStep(abc.ABC):
    @abc.abstractmethod
    def apply(self, data):
        # takes one argument; name is irrelevant
        pass

    def __call__(self, data=None):
        return self.load(data=data)


class Pipeline(PreprocessingStep):
    def __init__(self, steps):
        super().__init__()
        self.steps = steps

    def apply(self, data=None):
        out = data
        for step in self.steps:
            out = step.apply(out)

        return out


class ParallelPipeline(PreprocessingStep):
    def __init__(self, pipelines):
        super().__init__()
        self.pipelines = pipelines

    def apply(self, data):
        out = []
        for pipeline in self.pipelines:
            out.append(pipeline.apply(data))

        return list(zip(*out))


class FileRule(PreprocessingStep):
    def __init__(self, value, func="startswith"):
        super().__init__()
        self.value = value
        self.func = func

    def apply(self, file):
        func = getattr(file, self.func)
        return func(self.value)


class FileFinder(PreprocessingStep):
    def __init__(self, data_dir=None, rules=(), warn=True):
        super().__init__()
        self.data_dir = data_dir
        self.rules = rules
        self.warn = warn

    def apply(self, data=None):
        data_dir = data or self.data_dir

        files = os.listdir(data_dir)

        # TODO: also implement as a pipeline?
        for rule in self.rules:
            files = filter(rule.apply, files)

        out = list(map(lambda name: os.path.join(data_dir, name), files))

        if self.warn and len(out) == 0:
            warnings.warn(f"Couldn't find file in: {data_dir}")

        if len(out) == 1:
            return out[0]

        return out


class Path(PreprocessingStep):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def apply(self, data=None):
        return data or self.path


class PathShortener(PreprocessingStep):
    def __init__(self, init_index=-2, last_index=-1):
        self.init_index = init_index
        self.last_index = last_index

    def apply(self, path_name):
        path_ls = path_name.split(os.path.sep)
        return f"{os.path.sep}".join(path_ls[self.init_index : self.last_index])


class IndexSelector(PreprocessingStep):
    def __init__(self, index, repeat=False):
        super().__init__()
        self.index = index
        self.repeat = repeat

    def apply(self, data):
        selected = data[self.index]
        if self.repeat:
            return [selected] * len(data)

        return selected


class MriImageLoader(PreprocessingStep):
    def apply(self, filename):
        img = nib.load(filename)
        img_data = img.get_fdata()

        return img_data


class HashWithIncoming(PreprocessingStep):
    def __init__(self, step):
        super().__init__()
        self.step = step

    def apply(self, data):
        new_data = self.step.apply(data)

        return {datum: new_datum for datum, new_datum in zip(data, new_data)}


class TupleWithIncoming(PreprocessingStep):
    def __init__(self, step):
        super().__init__()
        self.step = step

    def apply(self, data):
        new_data = self.step.apply(data)
        return [(datum, new_datum) for datum, new_datum in zip(data, new_data)]


class Sorter(PreprocessingStep):
    def apply(self, data):
        return sorted(data)


class Filter(PreprocessingStep):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def apply(self, data):
        return list(filter(self.func, data))


class EmptyRemover(Filter):
    def __init__(self):
        super().__init__(func=lambda x: len(x) > 0)


class NoneRemover(Filter):
    def __init__(self):
        super().__init__(func=lambda x: x is not None)


class SerialMap(PreprocessingStep):
    def __init__(self, step, pbar=False):
        super().__init__()
        self.step = step
        self.pbar = pbar

    def apply(self, data):
        return [self.step.apply(datum) for datum in tqdm(data, disable=not self.pbar)]


class ParallelMap(PreprocessingStep):
    def __init__(self, step, n_jobs=-1, verbose=0):
        super().__init__()
        self.step = step
        self.n_jobs = n_jobs
        self.verbose = verbose

    def apply(self, data):
        with Parallel(n_jobs=self.n_jobs, verbose=self.verbose) as parallel:
            res = parallel(delayed(self.step.apply)(datum) for datum in data)

        return list(res)


class Map:
    def __new__(cls, step, n_jobs=0, verbose=0):
        if n_jobs != 0:
            return ParallelMap(step, n_jobs=n_jobs, verbose=verbose)

        return SerialMap(step, pbar=verbose > 0)


class Truncater(PreprocessingStep):
    # useful for debugging

    def __init__(self, value):
        super().__init__()
        self.value = value

    def apply(self, data):
        return data[: self.value]


class PdCsvReader(PreprocessingStep):
    def __init__(self, delimiter=","):
        super().__init__()
        self.delimiter = delimiter

    def apply(self, data):
        return pd.read_csv(data, delimiter=",")


class MeshExtractorFromSegmentedImage(PreprocessingStep):
    """Mesh extractor from images.

    Structure_ID names, numbers, and colors:
    ----------------------------------------
    1   255    0    0        1  1  1    "CA1"
    2     0  255    0        1  1  1    "CA2+3"
    3     0    0  255        1  1  1    "DG"
    4   255  255    0        1  1  1    "ERC"
    5     0  255  255        1  1  1    "PHC"
    6   255    0  255        1  1  1    "PRC"
    7    80  179  221        1  1  1    "SUB"
    8   255  215    0        1  1  1    "AntHipp"
    9   184  115   51        1  1  1    "PostHipp"
    2, 6 are expected to grow in volume with progesterone
    4, 5 are expected to shrink in volume with progesterone
    """

    def __init__(self, structure_id):
        super().__init__()
        self.structure_id = structure_id

    def apply(self, img_fdata):
        """Extract one surface mesh from the fdata of a segmented image.

        Parameters
        ----------
        img_fdata: array-like, shape = [n_x, n_y, n_z]. Voxels which are colored
            according to substructure assignment. For example, color of voxel
            (0, 0, 0) is an integer value that can be anywhere from 0-9.
        """
        # print(f"Img fdata shape: {img_fdata.shape}")
        if self.structure_id == -1:
            img_mask = img_fdata != 0
        else:
            img_mask = img_fdata == self.structure_id

        masked_img_fdata = np.where(img_mask, img_fdata, 0)
        # print(f"Masked img fdata shape: {masked_img_fdata.shape}")
        (
            vertices,
            faces,
            _,
            values,
        ) = skimage.measure.marching_cubes(  # omitted value is "normals"
            masked_img_fdata,
            level=0,
            step_size=1,
            allow_degenerate=False,
            method="lorensen",
        )
        # print(f"Colors: {values.min()}, {values.max()}")
        # print(f"Vertices.shape = {vertices.shape}")

        color_dict = {  # trimesh uses format [r, g, b, a] where a is alpha
            1: [255, 0, 0, 255],
            2: [0, 255, 0, 255],
            3: [0, 0, 255, 255],
            4: [255, 255, 0, 255],
            5: [0, 255, 255, 255],
            6: [255, 0, 255, 255],
            7: [80, 179, 221, 255],
            8: [255, 215, 0, 255],
            9: [184, 115, 51, 255],
        }

        colors = np.array([np.array(color_dict[value]) for value in values])
        # print(f"Colors.shape = {colors.shape}")
        # print("Colors:", colors)

        return vertices, faces, colors


class TrimeshFromData(PreprocessingStep):
    def apply(self, mesh):
        # TODO: make a check for colors?
        vertices, faces, colors = mesh
        return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)


class TrimeshToData(PreprocessingStep):
    def apply(self, mesh):
        return (
            np.array(mesh.vertices),
            np.array(mesh.faces),
            np.array(mesh.visual.vertex_colors),
        )


class MeshCenterer(PreprocessingStep):
    def apply(self, mesh):
        """Center a mesh by putting its barycenter at origin of the coordinates.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            Mesh to center.

        Returns
        -------
        centered_mesh : trimesh.Trimesh
            Centered Mesh.
        hippocampus_center: coordinates of center of the mesh before centering
        """
        vertices = mesh.vertices
        center = np.mean(vertices, axis=0)
        mesh.vertices = vertices - center

        return mesh


class MeshScaler(PreprocessingStep):
    def __init__(self, scaling_factor=20.0):
        super().__init__()
        self.scaling_factor = scaling_factor

    def apply(self, mesh):
        mesh.vertices = mesh.vertices / self.scaling_factor
        return mesh


class TrimeshFaceRemoverByArea(PreprocessingStep):
    # TODO: generalize?

    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def apply(self, mesh):
        face_mask = ~np.less(mesh.area_faces, self.threshold)
        mesh.update_faces(face_mask)

        return mesh


class TrimeshDegenerateFacesRemover(PreprocessingStep):
    """Trimesh degenerate faces remover.

    Parameters
    ----------
    height: float
        Identifies faces with an oriented bounding box shorter than
        this on one side.
    """

    def __init__(self, height=1e-08):
        self.height = height

    def apply(self, mesh):
        mesh.update_faces(mesh.nondegenerate_faces(height=self.height))
        return mesh


class TrimeshDecimator(PreprocessingStep):
    """Trimesh simplify quadratic decimation.

    Parameters
    ----------
    percent : float
        A number between 0.0 and 1.0 for how much.
    face_count : int
        Target number of faces desired in the resulting mesh.
    agression: int
        An integer between 0 and 10, the scale being roughly 0 is
        “slow and good” and 10 being “fast and bad.”
    """

    # NB: uses fast-simplification

    def __init__(self, percent=None, face_count=None, agression=None):
        super().__init__()
        self.percent = percent
        self.face_count = face_count
        self.agression = agression

    def apply(self, mesh):
        # TODO: issue with colors?
        decimated_mesh = mesh.simplify_quadric_decimation(
            percent=self.percent,
            face_count=self.face_count,
            aggression=self.agression,
        )

        # # TODO: delete
        # colors_ = np.array(decimated_mesh.visual.vertex_colors)
        # print("unique colors after decimation:", len(np.unique(colors_, axis=0)))

        return decimated_mesh


class FastSimplificationDecimator(PreprocessingStep):
    def __init__(self, target_reduction=0.25):
        super().__init__()
        self.target_reduction = target_reduction
        self._nbrs = NearestNeighbors(n_neighbors=1)

    def apply(self, mesh):
        # TODO: make a check for colors?
        vertices, faces, colors = mesh

        vertices_, faces_ = fast_simplification.simplify(
            vertices, faces, self.target_reduction
        )

        # TODO: can this be done better?
        self._nbrs.fit(vertices)
        _, indices = self._nbrs.kneighbors(vertices_)
        indices = np.squeeze(indices)
        colors_ = colors[indices]

        return vertices_, faces_, colors_


class H2MeshDecimator(PreprocessingStep):
    def __init__(self, decimation_factor=10.0):
        super().__init__()
        self.decimation_factor = decimation_factor

    def apply(self, mesh):
        # TODO: issues using due to open3d (delete?)

        # TODO: make a check for colors?
        vertices, faces, colors = mesh

        n_faces_after_decimation = faces.shape[0] // self.decimation_factor
        (
            vertices_after_decimation,
            faces_after_decimation,
            colors_after_decimation,
        ) = H2_SurfaceMatch.utils.utils.decimate_mesh(
            vertices, faces, n_faces_after_decimation, colors=colors
        )

        return [
            vertices_after_decimation,
            faces_after_decimation,
            colors_after_decimation,
        ]


class TrimeshToPly(PreprocessingStep):
    def __init__(self, dirname=""):
        self.dirname = dirname
        # TODO: create dir if does not exist?

        # TODO: add override?

    def apply(self, data):
        filename, mesh = data

        ext = ".ply"
        if not filename.endswith(ext):
            filename += ext

        path = os.path.join(self.dirname, filename)

        ply_text = trimesh.exchange.ply.export_ply(
            mesh, encoding="binary", include_attributes=True
        )

        # TODO: add verbose
        # print(f"- Write mesh to {filename}")
        with open(path, "wb") as file:
            file.write(ply_text)

        return data


class TrimeshReader(PreprocessingStep):
    # TODO: update
    def apply(self, path):
        return trimesh.load(path)


# TODO: create mesh serializer


class H2MeshAligner(PreprocessingStep):
    def __init__(
        self,
        a0=0.01,
        a1=10.0,
        b1=10.0,
        c1=1.0,
        d1=0.0,
        a2=1.0,
        resolutions=0,
        paramlist=(),
    ):
        super().__init__()
        self.a0 = a0
        self.a1 = a1
        self.b1 = b1
        self.c1 = c1
        self.d1 = d1
        self.a2 = a2
        self.resolutions = resolutions
        self.paramlist = paramlist

        # TODO: allow control of device
        self.device = None

    def apply(self, meshes):
        target_mesh, template_mesh = meshes

        # TODO: upgrade device?
        geod, F0, color0 = H2_SurfaceMatch.H2_match.H2MultiRes(
            source=template_mesh,
            target=target_mesh,
            a0=self.a0,
            a1=self.a1,
            b1=self.b1,
            c1=self.c1,
            d1=self.d1,
            a2=self.a2,
            resolutions=self.resolutions,
            start=None,
            paramlist=self.paramlist,
            device=self.device,
        )

        return geod[-1], F0, color0


class IdentityMeshAligner(PreprocessingStep):
    # useful for debugging

    def apply(self, meshes):
        target_mesh, _ = meshes
        return target_mesh


class Data:
    # TODO: delete?
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        # TODO: check
        if index == 0:
            return self.X

        elif index == 1:
            return self.y

        raise IndexError("Index can only be 1 or 2")


class ModelLoader(DataLoader):
    # TODO: create variants

    def __init__(self, data, model):
        self.data = data
        self.model = model

    def load(self):
        X, y = self.data

        # TODO: fit only if data?
        # TODO: wrap if pd?
        return self.model.fit(X, y=y)

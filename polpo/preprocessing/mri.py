import nibabel as nib
import numpy as np
import skimage

from polpo.utils import params_to_kwargs

from .base import PreprocessingStep

BRAINSTRUCT2COLOR = {
    "PRC": (255, 0, 255, 255),
    "PHC": (0, 255, 255, 255),
    "AntHipp": (255, 215, 0, 255),
    "ERC": (255, 255, 0, 255),
    "SUB": (80, 179, 221, 255),
    "PostHipp": (184, 115, 51, 255),
    "CA1": (255, 0, 0, 255),
    "DG": (0, 0, 255, 255),
    "CA2+3": (0, 255, 0, 255),
}


class MriImageLoader(PreprocessingStep):
    def apply(self, filename):
        img = nib.load(filename)
        img_data = img.get_fdata()

        return img_data


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

    def __init__(self, structure_id=-1, marching_cubes=None):
        if marching_cubes is None:
            marching_cubes = SkimageMarchingCubes(level=0, method="lewiner")

        super().__init__()
        self.structure_id = structure_id
        self.marching_cubes = marching_cubes

        self.color_dict = {  # trimesh uses format [r, g, b, a] where a is alpha
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

    def apply(self, img_fdata):
        """Extract one surface mesh from the fdata of a segmented image.

        Parameters
        ----------
        img_fdata: array-like, shape = [n_x, n_y, n_z]. Voxels which are colored
            according to substructure assignment. For example, color of voxel
            (0, 0, 0) is an integer value that can be anywhere from 0-9.
        """
        # TODO: implement as a pipeline (syntax sugar!)

        if self.structure_id == -1:
            img_mask = img_fdata != 0
        else:
            img_mask = img_fdata == self.structure_id

        masked_img_fdata = np.where(img_mask, img_fdata, 0)
        (
            vertices,
            faces,
            values,
        ) = self.marching_cubes(masked_img_fdata)

        colors = np.array([np.array(self.color_dict[value]) for value in values])
        return vertices, faces, colors


class SkimageMarchingCubes(PreprocessingStep):
    """
    https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes
    """

    def __init__(
        self,
        level=0,
        step_size=1,
        allow_degenerate=False,
        method="lewiner",
    ):
        self.level = level
        self.step_size = step_size
        self.allow_degenerate = allow_degenerate
        self.method = method

    def apply(self, data):
        if isinstance(data, tuple):
            img_fdata, mask = data
        else:
            img_fdata = data
            mask = None

        (
            vertices,
            faces,
            _,
            values,
        ) = skimage.measure.marching_cubes(
            img_fdata, mask=mask, **params_to_kwargs(self)
        )
        return vertices, faces, values

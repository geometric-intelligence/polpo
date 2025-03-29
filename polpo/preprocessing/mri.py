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
    """Load image from .nii file.

    Parameters
    ----------
    filename : str
        File to load.
    as_nib : bool
        Whether to return the nibabel object.
    return_affine : bool
        Whether to return the affine transformation.
        Ignore if `as_nib` is True.
    """

    def __init__(self, filename=None, as_nib=False, return_affine=False):
        super().__init__()
        self.filename = filename
        self.as_nib = as_nib
        self.return_affine = return_affine

    def apply(self, filename=None):
        """Apply step.

        Parameters
        ----------
        filename : str
            File to load.

        Returns
        -------
        img : nibabel.nifti1.Nifti1Image
            If `as_nib` is True.
        img_data : np.array
            If `as_nib` is False.
        affine : nibabel.nifti1.Nifti1Image
            If `as_nib` is False and `affine` is True.
        """
        filename = filename or self.filename

        img = nib.load(filename)
        if self.as_nib:
            return img

        img_data = img.get_fdata()
        if not self.return_affine:
            return img_data

        return img_data, img.affine


class LocalToTemplateTransform(PreprocessingStep):
    """Get transform from local to template system.

    Parameters
    ----------
    template_affine : array-like
        Target affine matrix.
    """

    def __init__(self, template_affine=None):
        super().__init__()
        self.template_affine = template_affine

        if self.template_affine is not None:
            self._template_affine_inv = np.linalg.inv(self.template_affine)

    def apply(self, data):
        """Apply step.

        Parameters
        ----------
        data : array-like or tuple[array-like; 2]
            (local, template) affine matrices.

        Returns
        -------
        transformation_matrix : np.array
            Transformation matrix.
        """
        if isinstance(data, (list, tuple)):
            local_affine, template_affine = data
            template_affine_inv = np.linalg.inv(template_affine)

        else:
            if self.template_affine is None:
                raise ValueError("Template affine is undefined.")
            template_affine_inv = self._template_affine_inv

        return template_affine_inv @ local_affine


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

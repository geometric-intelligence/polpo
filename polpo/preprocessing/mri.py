import nibabel as nib
import numpy as np
import skimage

from polpo.utils import params_to_kwargs

from .base import PreprocessingStep

HippStructs = {
    "CA1",
    "CA2+3",
    "DG",
    "ERC",
    "PHC",
    "PRC",
    "SUB",
    "AntHipp",
    "PostHipp",
}


class StructEncoding:
    def __init__(self, structs, ids, colors):
        self.structs = structs
        self.ids = ids
        self.colors = colors

        self._structs2ids = None
        self._ids2colors = None

    @property
    def structs2ids(self):
        if self._structs2ids is None:
            self._structs2ids = dict(zip(self.structs, self.ids))

        return self._structs2ids

    @property
    def ids2colors(self):
        if self._ids2colors is None:
            self._ids2colors = dict(zip(self.ids, self.colors))

        return self._ids2colors


class AshsPrincetonYoungAdult3TEncoding(StructEncoding):
    """Encoding of hippocampus subfields.

    https://www.nitrc.org/projects/ashs

    Check out [PTC2024]_ for more details.

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

    References
    ----------
    .. [PTC2024] L. Pritschet, C.M. Taylor, et al., 2024. Neuroanatomical changes observed
        over the course of a human pregnancy. Nat Neurosci 27, 2253–2260.
        https://doi.org/10.1038/s41593-024-01741-0
    """

    def __init__(self):
        structs = [
            "CA1",
            "CA2+3",
            "DG",
            "ERC",
            "PHC",
            "PRC",
            "SUB",
            "AntHipp",
            "PostHipp",
        ]
        labels = (1, 2, 3, 4, 5, 6, 7, 8, 9)
        colors = [
            (255, 0, 0, 255),
            (0, 255, 0, 255),
            (0, 0, 255, 255),
            (255, 255, 0, 255),
            (0, 255, 255, 255),
            (255, 0, 255, 255),
            (80, 179, 221, 255),
            (255, 215, 0, 255),
            (184, 115, 51, 255),
        ]
        super().__init__(structs, labels, colors)


class BrainStructEncoding(StructEncoding):
    def __init__(
        self, midline_structs, bilateral_structs, ids, midline_colors, bilateral_colors
    ):
        self.midline_structs = midline_structs
        self.bilateral_structs = bilateral_structs

        sides = ["L", "R"]

        structs = midline_structs + [
            f"{side}_{struct}" for struct in bilateral_structs for side in sides
        ]

        colors = midline_colors.copy()
        for color in bilateral_colors:
            colors.extend([color, color])

        super().__init__(structs, ids, colors)


class FreeSurferStructEncoding(BrainStructEncoding):
    """
    https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
    """

    def __init__(self):
        midline_structs = ["BrStem"]
        bilateral_structs = [
            "Thal",
            "Caud",
            "Puta",
            "Pall",
            "Hipp",
            "Amyg",
            "Accu",
        ]

        ids = [
            16,
            10,  # Left-Thamalus-Proper
            49,  # Right-Thamalus-Proper
            11,
            50,
            12,
            51,
            13,
            52,
            17,
            53,
            18,
            54,
            26,
            58,
        ]

        # NB: sides have same color
        midline_colors = [(119, 159, 176, 0)]
        bilateral_colors = [
            (0, 118, 14, 0),
            (122, 186, 220, 0),
            (236, 13, 176, 0),
            (13, 48, 255, 0),
            (220, 216, 20, 0),
            (103, 255, 255, 0),
            (255, 165, 0, 0),
        ]

        super().__init__(
            midline_structs, bilateral_structs, ids, midline_colors, bilateral_colors
        )


class FslFirstStructEncoding(FreeSurferStructEncoding):
    """
    https://fsl.fmrib.ox.ac.uk/fsl/docs/#/structural/first?id=supported-structures

    NB: compatible with FreeSurfer for the existing ones.
    """


def segmtool2encoding(tool=None, struct=None, raise_=False):
    # struct is ignored if tool is not None

    if tool is None and isinstance(struct, str):
        if struct in HippStructs:
            return AshsPrincetonYoungAdult3TEncoding()

        return FreeSurferStructEncoding()

    if tool is None:
        if raise_:
            raise ValueError("Need to know tool or struct")

        return None

    tool_l = tool.lower()
    if tool_l.startswith("ashs"):
        return AshsPrincetonYoungAdult3TEncoding()

    if tool_l.startswith("fast") or tool_l.startswith("free"):
        return FreeSurferStructEncoding()

    if tool_l.startswith("fsl"):
        return FslFirstStructEncoding()

    if raise_:
        raise ValueError(f"Cannot manage `{tool}`")


class MriImageLoader(PreprocessingStep):
    """Load image from .nii/.mgz file.

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

    def __call__(self, filename=None):
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

    def __call__(self, data):
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

    Parameters
    ----------
    struct_id : int or str
        Structure to select. Depends on segmentation tool.
        If -1 considers full structure. If integer, ignores encoding.
    marching_cubes : callable
        Marching cubes algorithm.
    return_colors : bool
        Whether to output colors.
        Ignored if marching_cubes is not None or no encoding.
        For the former, output len sets value.
    encoding : str or StructEncoding
        Encoding or used segmentation tool.
        One of the following: 'ashs*', 'fast*', 'free*', 'fsl*'.
        If None, tries to figure it out.
    """

    def __init__(
        self,
        struct_id=-1,
        marching_cubes=None,
        return_colors=True,
        encoding=None,
    ):
        if encoding is None or isinstance(encoding, str):
            encoding = segmtool2encoding(encoding, struct_id)

        if encoding is None and isinstance(struct_id, str):
            raise ValueError(f"Need encoding to handle str id: `{struct_id}`")

        if marching_cubes is None:
            marching_cubes = SkimageMarchingCubes(
                level=0, method="lewiner", return_values=return_colors
            )

        super().__init__()
        self.structure_id = struct_id
        self.marching_cubes = marching_cubes
        self.encoding = encoding

    def __call__(self, img_fdata):
        """Extract one surface mesh from the fdata of a segmented image.

        Parameters
        ----------
        img_fdata: array-like, shape = [n_x, n_y, n_z]. Voxels which are colored
            according to substructure assignment. For example, color of voxel
            (0, 0, 0) is an integer value that can be anywhere from 0-9.
        """
        structure_id = (
            self.structure_id
            if isinstance(self.structure_id, int)
            else self.encoding.structs2ids.get(self.structure_id)
        )

        if structure_id == -1:
            img_mask = img_fdata != 0
        else:
            img_mask = img_fdata == structure_id

        masked_img_fdata = np.where(img_mask, img_fdata, 0)
        # (vertices, faces) or (vertices, faces, values)
        out = self.marching_cubes(masked_img_fdata)

        if len(out) == 2:
            return out

        if self.encoding is None:
            return out[:2]

        colors2dict = self.encoding.ids2colors
        colors = np.array([np.array(colors2dict[value]) for value in out[-1]])
        return out[:2] + (colors,)


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
        return_normals=False,
        return_values=False,
    ):
        super().__init__()
        self.level = level
        self.step_size = step_size
        self.allow_degenerate = allow_degenerate
        self.method = method
        self.return_normals = return_normals
        self.return_values = return_values

    def __call__(self, data):
        if isinstance(data, tuple):
            img_fdata, mask = data
        else:
            img_fdata = data
            mask = None

        (
            vertices,
            faces,
            normals,
            values,
        ) = skimage.measure.marching_cubes(
            img_fdata,
            mask=mask,
            **params_to_kwargs(self, ignore=("return_values", "return_normals")),
        )

        out = (vertices, faces)

        if self.return_normals:
            out = out + (normals,)

        if self.return_values:
            out = out + (values,)

        return out

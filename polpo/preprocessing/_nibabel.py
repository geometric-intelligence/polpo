import nibabel

from polpo.preprocessing.base import PreprocessingStep


class FreeSurferReader(PreprocessingStep):
    def __call__(self, filename):
        """Apply step.

        Parameters
        ----------
        filename : str
            File name.

        Returns
        -------
        vertices : np.array
            Mesh vertices.
        faces : np.array
            Mesh faces.
        """
        return nibabel.freesurfer.read_geometry(filename)

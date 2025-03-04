import numpy as np
import trimesh

from polpo.preprocessing.base import PreprocessingStep

class GetMeshFromImage(PreprocessingStep):
    def __init__(self, image=None):
        super().__init__()
        self.image = image  

    def apply(self, image=None):
        """
        Extracts 'mesh' from the image dictionary and returns it as a list.
        """
        image = image if image is not None else self.image

        if image and "mesh" in image:
            return {"template_mesh": image["mesh"]}
        else:
            raise ValueError("Key 'mesh' not found in image dictionary.")

class ComputeBounds(PreprocessingStep):
    def __init__(self, meshes=None):
        super().__init__()
        self.meshes = meshes

    def apply(self, data=None):
        """
        Compute the bounding box of a given dictionary of meshes.

        Args:
            data: The input data (e.g., meshes).

        Returns:
            mins: numpy array of minimum values for each axis.
            maxs: numpy array of maximum values for each axis.
        """
        if data is not None:
            # Use the data passed in the argument
            self.meshes = data

        """Compute the bounding box for the union of multiple meshes."""
        if not self.meshes:
            return np.array([-1, -1, -1]), np.array([1, 1, 1])

        all_vertices = np.vstack([mesh.vertices for mesh in list(self.meshes.values())])  # Combine all mesh vertices
        mins = np.min(all_vertices, axis=0)
        maxs = np.max(all_vertices, axis=0)

        # self.bounds = (mins, maxs) 
        return mins, maxs
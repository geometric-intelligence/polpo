from pathlib import Path

# TODO: rename to paths
# TODO: rename to LddmmPaths


class DirConfig:
    def __init__(
        self,
        outputs_dir,
        meshes_dir=None,
        registrations_dir=None,
        transports_dir=None,
        shoots_dir=None,
        atlases_dir=None,
    ):
        self.outputs_dir = outputs_dir

        self.meshes_dir = self._resolve(meshes_dir or "meshes")
        self.registrations_dir = self._resolve(registrations_dir or "registrations")
        self.transports_dir = self._resolve(transports_dir or "transports")
        self.shoots_dir = self._resolve(shoots_dir or "shoots")
        self.atlases_dir = self._resolve(atlases_dir or "atlases")

    def _resolve(self, path):
        path = Path(path)

        if path.is_absolute():
            return path

        return self.outputs_dir / path

    def resolve(self, path):
        """Resolve a bundle-relative artifact path."""
        return self._resolve(path)

    def relative(self, path):
        """Convert an artifact path to a bundle-relative path."""
        return path.relative_to(self.outputs_dir)

    def to_dict(self):
        return {
            "meshes_dir": self.relative(self.meshes_dir).as_posix(),
            "registrations_dir": self.relative(self.registrations_dir).as_posix(),
            "transports_dir": self.relative(self.transports_dir).as_posix(),
            "shoots_dir": self.relative(self.shoots_dir).as_posix(),
            "atlases_dir": self.relative(self.atlases_dir).as_posix(),
        }

    @classmethod
    def from_dict(cls, outputs_dir, data):
        return cls(
            outputs_dir=outputs_dir,
            **data,
        )

    def registration_path(self, id_):
        return self.registrations_dir / id_

    def transport_path(self, id_):
        return self.transports_dir / id_

    def shoot_path(self, id_):
        return self.shoots_dir / id_

    def atlas_path(self, id_):
        return self.atlases_dir / id_

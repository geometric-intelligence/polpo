class DirConfig:
    def __init__(
        self,
        outputs_dir,
        meshes_dir=None,
        registration_dir=None,
        transport_dir=None,
        shoot_dir=None,
        atlas_dir=None,
    ):
        if meshes_dir is None:
            meshes_dir = outputs_dir / "meshes"

        if registration_dir is None:
            registration_dir = outputs_dir / "registrations"

        if transport_dir is None:
            transport_dir = outputs_dir / "transports"

        if shoot_dir is None:
            shoot_dir = outputs_dir / "shoots"

        if atlas_dir is None:
            atlas_dir = outputs_dir / "atlases"

        self.outputs_dir = outputs_dir
        self.meshes_dir = meshes_dir
        self.registration_dir = registration_dir
        self.shoot_dir = shoot_dir
        self.atlas_dir = atlas_dir
        self.transport_dir = transport_dir

        self.meshes_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self):
        return {
            "meshes_dir": self.meshes_dir.relative_to(self.outputs_dir).as_posix(),
            "registration_dir": self.registration_dir.relative_to(
                self.outputs_dir
            ).as_posix(),
            "transport_dir": self.transport_dir.relative_to(
                self.outputs_dir
            ).as_posix(),
            "shoot_dir": self.shoot_dir.relative_to(self.outputs_dir).as_posix(),
            "atlas_dir": self.atlas_dir.relative_to(self.outputs_dir).as_posix(),
        }

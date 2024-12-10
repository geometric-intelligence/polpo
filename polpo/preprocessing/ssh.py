import os

import paramiko
import scp

from polpo.defaults import DATA_DIR

from .base import CacheableDataLoader, PreprocessingStep


def connect_using_ssh_config(host_name):
    """Connect to a host using SSH config."""
    config = paramiko.SSHConfig()
    with open(os.path.expanduser("~/.ssh/config")) as file:
        config.parse(file)

    host_config = config.lookup(host_name)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    client.connect(
        hostname=host_config.get("hostname", host_name),
        port=host_config.get("port", 22),
        username=host_config.get("user"),
        key_filename=host_config.get("identityfile"),
    )

    return client


class SCPDataLoader(PreprocessingStep, CacheableDataLoader):
    """

    Parameters
    ----------
    data_dir : str
        Directory where to store data.
    """

    def __init__(
        self,
        remote_path,
        data_dir=None,
        recursive=True,
        use_cache=True,
        ssh_client=None,
        host_name=None,
    ):
        # host_name is ignored if ssh_client
        # host_name is lazy
        # TODO: recursive to None and check if has extension?
        super().__init__(use_cache)

        if data_dir is None:
            data_dir = DATA_DIR

        if "~" in data_dir:
            data_dir = os.path.expanduser(data_dir)

        self._ssh_client = ssh_client if ssh_client is not None else host_name
        self.remote_path = remote_path
        self.recursive = recursive
        self.data_dir = data_dir

    @property
    def _path(self):
        remote_path_ls = self.remote_path.split(os.path.sep)
        return os.path.join(self.data_dir, remote_path_ls[-1])

    @property
    def ssh_client(self):
        if isinstance(self._ssh_client, str):
            self._ssh_client = connect_using_ssh_config(self._ssh_client)
        return self._ssh_client

    @classmethod
    def from_ssh_host(
        cls, host_name, remote_path, data_dir=None, recursive=True, use_cache=True
    ):
        ssh_client = connect_using_ssh_config(host_name)

        return cls(
            remote_path,
            ssh_client=ssh_client,
            data_dir=data_dir,
            recursive=recursive,
            use_cache=use_cache,
        )

    def load(self):
        path = self._path

        if self.exists(path):
            return path

        os.makedirs(self.data_dir, exist_ok=True)
        with scp.SCPClient(self.ssh_client.get_transport()) as scp_client:
            scp_client.get(
                remote_path=self.remote_path,
                local_path=self.data_dir,
                recursive=self.recursive,
            )

        return path

    def apply(self, data=None):
        return self.load()

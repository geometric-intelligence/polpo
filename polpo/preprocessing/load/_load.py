import json
import os
import shutil
import urllib
import warnings
import zipfile

import requests

from polpo.defaults import DATA_DIR
from polpo.preprocessing.base import CacheableDataLoader, PreprocessingStep
from polpo.preprocessing.path import ExpandUser


def _get_basename(path):
    return path.split(os.path.sep)[-1]


class FigshareDataLoader(PreprocessingStep, CacheableDataLoader):
    """Transfer files and directories from figshare.

    Parameters
    ----------
    figshare_id : int
        Id of figshare article.
    remote_path : str
        Path to retrieve from remote host.
    data_dir : str
        Directory where to store data.
    use_cache : bool
        Whether to verify if data is already available locally.
    local_basename : str
        Basename of transferred file/folder if different from remote host.
    version : int
        Dataset version.
    remove_id : bool
        Whether to remove figshare added id when downloading items that are
        within a folder.

    Notes
    -----
    * If ``remote_path`` does not exist in figshare, it automatically downloads
    the full dataset. Code will fail after download. Warning will give information
    about downloaded zip. Can't know in advance if this will happen.
    """

    def __init__(
        self,
        figshare_id,
        remote_path,
        data_dir=None,
        use_cache=True,
        local_basename=None,
        version=1,
        remove_id=True,
    ):
        super().__init__(use_cache)

        if data_dir is None:
            data_dir = DATA_DIR

        self.figshare_id = figshare_id
        self.remote_path = remote_path
        self.data_dir = ExpandUser()(data_dir)
        self.local_basename = local_basename
        self.version = version
        self.remove_id = remove_id

        self._base_api_url = "https://api.figshare.com/v2"
        self._base_download_url = "https://figshare.com/ndownloader"

    @property
    def _path(self):
        return os.path.join(self.data_dir, _get_basename(self.remote_path))

    @property
    def _renamed_path(self):
        if self.local_basename is None:
            return self._path

        return os.path.join(self.data_dir, self.local_basename)

    def _build_api_url(self):
        return (
            f"{self._base_api_url}/articles/{self.figshare_id}/versions/{self.version}"
        )

    def _build_download_url(self):
        return f"{self._base_download_url}/articles/{self.figshare_id}/versions/{self.version}"

    def _load_file(self):
        url = self._build_api_url()

        response = requests.get(url)
        response.raise_for_status()

        # figshare does not provide full path for file
        filename = _get_basename(self.remote_path)
        metadata = json.loads(response.text)

        files_metadata = list(
            filter(lambda x: x["name"] == filename, metadata["files"])
        )
        if len(files_metadata) > 1:
            raise ValueError(
                "Can't disambiguate filename, please download folder instead"
            )
        elif len(files_metadata) == 0:
            raise ValueError("File does not exist, please check for typos.")

        url = files_metadata[0]["download_url"]

        os.makedirs(self.data_dir, exist_ok=True)
        filename, _ = urllib.request.urlretrieve(url, self._renamed_path)

        return filename

    def _load_folder(self):
        query_params = urllib.parse.urlencode({"folder_path": self.remote_path})

        url = self._build_download_url()
        url += f"?{query_params}"

        zip_filename, _ = urllib.request.urlretrieve(url)

        renamed_path = self._renamed_path
        extraction_path = os.path.join(renamed_path, "_tmp")
        os.makedirs(extraction_path, exist_ok=True)

        full_extraction_path = os.path.join(extraction_path, self.remote_path)

        # to keep folder structure
        index_middle = len(self.remote_path.split(os.path.sep))

        try:
            with zipfile.ZipFile(zip_filename, "r") as folder:
                for file in folder.filelist:
                    if not file.filename.startswith(self.remote_path):
                        continue

                    folder.extract(file, extraction_path)

                    basename = _get_basename(file.filename)
                    new_basename = (
                        basename[basename.index("_") + 1 :]
                        if (
                            self.remove_id
                            and "_" in basename
                            and basename[
                                : basename.index("_")
                            ].isdigit()  # figshare is inconsistent
                        )
                        else basename
                    )
                    middle_path_ls = file.filename.split(os.path.sep)[index_middle:-1]
                    middle_path = (
                        os.path.join(*middle_path_ls) if middle_path_ls else ""
                    )

                    if middle_path:
                        os.makedirs(
                            os.path.join(renamed_path, middle_path), exist_ok=True
                        )

                    os.rename(
                        os.path.join(full_extraction_path, middle_path, basename),
                        os.path.join(renamed_path, middle_path, new_basename),
                    )

                shutil.rmtree(extraction_path)

            os.remove(zip_filename)

        except Exception as e:
            warnings.warn(f"Can't handle zipfile: {zip_filename}")
            raise e

        return renamed_path

    def load(self):
        path = self._renamed_path

        if self.exists(path):
            return path

        is_file = "." in _get_basename(self.remote_path)

        if is_file:
            return self._load_file()

        return self._load_folder()

    def __call__(self, data=None):
        return self.load()

from abc import ABC, abstractmethod

from polpo.io.json import load_json, save_json
from polpo.time import Timer, utc_now


class TaskRunner(ABC):
    MANIFEST_VERSION = 1

    def __init__(self, state_dir, metadata=None):
        self.state_dir = state_dir
        self.metadata = metadata or None
        self.resolved_ = {}
        self.timer = Timer()

    @property
    def manifest_path(self):
        return self.state_dir / "manifest.json"

    @abstractmethod
    def tasks(self):
        """Return a mapping from task names to callables."""

    def set_resolved(self, **values):
        self.resolved_.update(values)

    def _new_manifest(self):
        return {
            "version": self.MANIFEST_VERSION,
            "metadata": self.metadata,
            "resolved": self.resolved_,
            "status": "running",
            "started_at": utc_now(),
            "tasks": {},
        }

    def _load_or_create_manifest(self):
        if self.manifest_path.exists():
            return load_json(self.manifest_path)

        return self._new_manifest()

    def _write_manifest(self):
        self.manifest_["resolved"] = dict(self.resolved_)
        self.manifest_["updated_at"] = utc_now()
        save_json(self.manifest_path, self.manifest_)

    def _is_complete(self, task):
        task_info = self.manifest_["tasks"].get(task, {})
        return task_info.get("status") == "completed"

    def _mark_completed(self, task):
        self.manifest_["tasks"][task] = {
            "status": "completed",
            "finished_at": utc_now(),
            "elapsed": self.timer.as_dict()[task],
        }

    def _mark_failed(self, task, error):
        self.manifest_["status"] = "failed"
        self.manifest_["tasks"][task] = {
            "status": "failed",
            "finished_at": utc_now(),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }

    def run(self, tasks=None, overwrite=False, continue_on_error=True):
        available_tasks = self.tasks()

        if tasks is None:
            tasks = list(available_tasks)

        unknown = set(tasks) - set(available_tasks)
        if unknown:
            raise ValueError(f"Unknown tasks: {sorted(unknown)}")

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_ = self._load_or_create_manifest()

        n_failed = 0
        for task in tasks:
            if not overwrite and self._is_complete(task):
                continue

            try:
                with self.timer(task):
                    available_tasks[task]()
            except Exception as error:
                n_failed += 1
                self._mark_failed(task, error)
                self._write_manifest()
                if not continue_on_error:
                    raise

                continue

            self._mark_completed(task)
            self._write_manifest()

        if n_failed == 0:
            status = "completed"
        elif n_failed == len(tasks):
            status = "failed"
        else:
            status = "partial"

        self.manifest_["status"] = status
        self.manifest_["finished_at"] = utc_now()
        self._write_manifest()

        return self

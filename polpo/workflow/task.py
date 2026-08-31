from abc import ABC

from polpo.io.json import load_json, save_json
from polpo.time import Timer, utc_now


def task(func):
    func._is_task = True
    return func


class TaskRunner(ABC):
    MANIFEST_VERSION = 1

    def __init__(self, state_dir, metadata=None, verbose=True):
        self.state_dir = state_dir
        self.metadata = metadata or None
        self.resolved_ = {}
        self.timer = Timer()
        self.verbose = verbose

    @property
    def manifest_path(self):
        return self.state_dir / "manifest.json"

    def tasks(self):
        tasks = {}

        for cls in reversed(type(self).mro()):
            for name, attr in cls.__dict__.items():
                if getattr(attr, "_is_task", False):
                    tasks[name] = getattr(self, name)

        return tasks

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

    def _log(self, message):
        if self.verbose:
            print(message)

    def _resolve_tasks(self, tasks, exclude_tasks):
        available_tasks = self.tasks()

        if tasks is None:
            tasks = list(available_tasks)

        exclude_tasks = [] if exclude_tasks is None else exclude_tasks

        unknown = (set(tasks) | set(exclude_tasks)) - set(available_tasks)
        if unknown:
            raise ValueError(f"Unknown tasks: {sorted(unknown)}")

        return {
            task: available_tasks[task] for task in tasks if task not in exclude_tasks
        }

    def _run_tasks(self, tasks, overwrite, continue_on_error):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_ = self._load_or_create_manifest()

        self._log(f"Running {self.__class__.__name__}")
        self._log(f"State directory: {self.state_dir}")

        n_failed = 0
        for task_name, task in tasks.items():
            if not overwrite and self._is_complete(task_name):
                self._log(f"[{task_name}] skipped (already completed)")
                continue

            self._log(f"[{task_name}] started")

            try:
                with self.timer(task_name):
                    task()
            except Exception as error:
                n_failed += 1
                self._mark_failed(task_name, error)
                self._write_manifest()

                self._log(f"[{task_name}] failed: {error}")

                if not continue_on_error:
                    raise

                continue

            elapsed = self.timer.duration(task_name)

            self._mark_completed(task_name)
            self._write_manifest()

            self._log(f"[{task_name}] completed in {elapsed:.2f} s")

        if n_failed == 0:
            status = "completed"
        elif n_failed == len(tasks):
            status = "failed"
        else:
            status = "partial"

        self.manifest_["status"] = status
        self.manifest_["finished_at"] = utc_now()
        self._write_manifest()

        self._log(f"{self.__class__.__name__} finished: {status}")

        return self

    def run(
        self,
        tasks=None,
        exclude_tasks=None,
        overwrite=False,
        continue_on_error=True,
    ):
        tasks = self._resolve_tasks(tasks, exclude_tasks)
        return self._run_tasks(tasks, overwrite, continue_on_error)


class CompositeTaskRunner(TaskRunner):
    def __init__(self, runners, state_dir, metadata=None, verbose=True):
        super().__init__(state_dir, metadata=metadata, verbose=verbose)
        self.runners = runners

    def tasks(self):
        return self.runners

    def _make_runner_task(
        self,
        runner,
        overwrite=False,
        continue_on_error=False,
    ):
        def run():
            runner.run(
                overwrite=overwrite,
                continue_on_error=continue_on_error,
            )

            if runner.manifest_["status"] != "completed":
                raise RuntimeError(
                    f"{runner.__class__.__name__} finished with "
                    f"status {runner.manifest_['status']!r}"
                )

        return run

    def run(
        self,
        tasks=None,
        exclude_tasks=None,
        overwrite=False,
        continue_on_error=False,
    ):
        tasks = self._resolve_tasks(tasks, exclude_tasks)

        tasks = {
            name: self._make_runner_task(
                runner,
                overwrite=overwrite,
                continue_on_error=continue_on_error,
            )
            for name, runner in tasks.items()
        }

        return self._run_tasks(
            tasks,
            overwrite=overwrite,
            continue_on_error=continue_on_error,
        )

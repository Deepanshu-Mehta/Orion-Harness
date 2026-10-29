"""Day 2 — LocalEnv: direct file operations and processes on this machine.

Concept: the baseline environment. Files go through `os`, commands through
`subprocess`, both bounded only by the path jail and process hygiene.

Design rules:
- Every file operation re-resolves its path, so the jail holds even if a
  caller skipped the gate.
- Writes go to a temp file in the same directory and are renamed into
  place, so a reader never sees half a file.
- `describe()` says plainly that commands are not isolated: the jail binds
  the file tools, not what a shell command can touch.
- Subclasses change where commands run by overriding `_wrap`, and where the
  workspace appears by setting `root` apart from `host_root`.
"""

import os
import re
import tempfile

from orion.env.base import (IGNORE_DIRS, Environment, ExecResult, JobTable,
                            glob_match, run_process)

_SEARCH_MAX_BYTES = 2 * 1024 * 1024
_SCAN_LIMIT = 100_000


class LocalEnv(Environment):
    """Run on this machine, in `root`, with no isolation."""

    name = "local"
    isolation = "none"
    exec_network = True

    def __init__(self, root: str):
        """Use `root` (which must exist) as the workspace."""
        if not os.path.isdir(root):
            raise FileNotFoundError(f"workspace root {root!r} is not a directory")
        self.host_root = os.path.realpath(root)
        self.root = self.host_root
        self._jobs = JobTable(os.path.join(self.host_root, ".orion", "jobs"))

    # Path mapping between the env's view (`root`) and this machine (`host_root`).
    def _to_host(self, path: str) -> str:
        if path == self.root or path.startswith(self.root.rstrip("/") + "/"):
            return self.host_root + path[len(self.root):]
        return path

    def _to_view(self, path: str) -> str:
        if path == self.host_root or path.startswith(self.host_root + os.sep):
            return self.root + path[len(self.host_root):]
        return path

    def _canonical(self, path: str) -> str:
        path = os.path.expanduser(path)
        if not os.path.isabs(path):
            path = os.path.join(self.root, path)
        # realpath flattens `..` and follows symlinks, so a link inside the
        # workspace that points outside it is caught by the prefix check.
        return self._to_view(os.path.realpath(self._to_host(os.path.normpath(path))))

    def _host_path(self, path: str) -> str:
        return self._to_host(self.resolve(path))

    def read_bytes(self, path: str) -> bytes:
        """Read a file inside the workspace."""
        with open(self._host_path(path), "rb") as f:
            return f.read()

    def write_bytes(self, path: str, data: bytes) -> None:
        """Write a file atomically, creating parent directories."""
        target = self._host_path(path)
        parent = os.path.dirname(target)
        os.makedirs(parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=parent, prefix=".orion-tmp-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            if os.path.exists(target):
                os.chmod(tmp, os.stat(target).st_mode & 0o7777)
            os.replace(tmp, target)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def exists(self, path: str) -> bool:
        """Whether the path exists inside the workspace."""
        return os.path.exists(self._host_path(path))

    def is_dir(self, path: str) -> bool:
        """Whether the path is a directory inside the workspace."""
        return os.path.isdir(self._host_path(path))

    def list_files(self, limit: int) -> list[str]:
        """Relative file paths under the root, ignore set applied, sorted."""
        found = []
        for dirpath, dirnames, filenames in os.walk(self.host_root):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
            rel_dir = os.path.relpath(dirpath, self.host_root)
            for name in filenames:
                found.append(name if rel_dir == "." else f"{rel_dir}/{name}".replace(os.sep, "/"))
        return sorted(found)[:limit]

    def search(self, regex: str, glob: str, limit: int, exclude=None) -> list[tuple[str, int, str]]:
        """Lines matching `regex` in text files matching `glob`."""
        pattern = re.compile(regex)
        hits = []
        for rel in self.list_files(_SCAN_LIMIT):
            if not glob_match(rel, glob) or (exclude and exclude(rel)):
                continue
            host = os.path.join(self.host_root, rel)
            try:
                if os.path.getsize(host) > _SEARCH_MAX_BYTES:
                    continue
                with open(host, "rb") as f:
                    data = f.read()
            except OSError:
                continue
            if b"\0" in data[:8192]:
                continue
            for lineno, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
                if pattern.search(line):
                    hits.append((rel, lineno, line))
                    if len(hits) >= limit:
                        return hits
        return hits

    def _wrap(self, command: str, timeout: float | None) -> list[str]:
        """The argv that runs `command`; subclasses add their jail here."""
        return ["/bin/sh", "-c", command]

    def _extra_env(self) -> dict:
        """Variables beyond the allowlist that the wrapper process needs."""
        return {}

    def exec(self, command: str, timeout: float) -> ExecResult:
        """Run a shell command in the workspace."""
        return run_process(self._wrap(command, timeout), cwd=self.host_root,
                           timeout=timeout, extra_env=self._extra_env())

    def spawn(self, command: str) -> str:
        """Start a background job; its log lives under `.orion/jobs/`."""
        job_id = self._jobs.new_id()
        return self._jobs.start(job_id, self._wrap_job(command, job_id), self.host_root,
                                command, self._extra_env())

    def _wrap_job(self, command: str, job_id: str) -> list[str]:
        return self._wrap(command, None)

    def job_output(self, job_id: str, tail_chars: int) -> tuple[str, bool]:
        """Return (tail of the job's log, still_running)."""
        return self._jobs.output(job_id, tail_chars)

    def job_stop(self, job_id: str) -> str:
        """Stop a job's process group."""
        return self._jobs.stop(job_id)

    def list_jobs(self) -> list[tuple[str, str, bool]]:
        """Return (id, command, running) for every job."""
        return self._jobs.list()

    def describe(self) -> str:
        """One line naming the environment and its (lack of) isolation."""
        return (f"local: commands run directly on this machine in {self.root} with NO "
                "isolation; a shell command can read and write anything this user can, "
                "and reach the network.")

    def close(self) -> None:
        """Stop all background jobs."""
        self._jobs.stop_all()

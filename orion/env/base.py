"""Day 2 — the operations layer: where a tool actually runs.

Concept: tools never call `open`, `os` or `subprocess` themselves. They call
an Environment, so the same `read_file` works on the laptop, inside an OS
sandbox, in a container, or on a remote host. The environment also decides
how far a mistake can reach.

Design rules:
- `resolve` is the path jail: every path is canonicalized and must land at
  or under the workspace root, or the operation raises PermissionError.
- Every exec gets the same process hygiene: a scrubbed environment (no
  secrets), stdin from /dev/null, its own process group, output to a temp
  file (not a pipe), a kill on timeout that still returns partial output,
  and a head+tail output cap.
- Background jobs are tracked, logged to a file, and stopped on close and
  at interpreter exit.
"""

import atexit
import functools
import os
import re
import signal
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass

IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".orion",
               "dist", "build", ".next"}

OUTPUT_CAP = 30_000
HEAD_KEEP = 10_000
TAIL_KEEP = 20_000
# Above this many bytes, only the head and tail of a log are read from disk.
_READ_ALL_LIMIT = 10 * 1024 * 1024

ENV_ALLOWLIST = ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "TMPDIR", "SHELL")
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


@dataclass
class ExecResult:
    """The outcome of one command."""

    output: str             # stdout+stderr combined, already capped
    exit_code: int | None   # None when killed by timeout
    timed_out: bool
    truncated: bool


def cap_output(text: str) -> tuple[str, bool]:
    """Cap text at OUTPUT_CAP chars, keeping the head and the (error-rich) tail."""
    if len(text) <= OUTPUT_CAP:
        return text, False
    dropped = len(text) - HEAD_KEEP - TAIL_KEEP
    return f"{text[:HEAD_KEEP]}\n... [{dropped} chars truncated] ...\n{text[-TAIL_KEEP:]}", True


def scrubbed_env(extra: dict | None = None) -> dict:
    """Build the environment for child processes: an allowlist, never secrets."""
    env = {k: os.environ[k] for k in ENV_ALLOWLIST if k in os.environ}
    env.update(extra or {})
    env.update({"PYTHONUNBUFFERED": "1", "CI": "1"})
    # The name check runs last so nothing, including `extra`, can smuggle a
    # secret-looking variable into a command.
    return {k: v for k, v in env.items()
            if not any(marker in k.upper() for marker in SECRET_MARKERS)}


@functools.lru_cache(maxsize=512)
def _glob_regex(pattern: str) -> re.Pattern:
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def glob_match(path: str, pattern: str) -> bool:
    """Match a slash-separated path against a glob.

    `**/` spans any number of directories, `*` and `?` stay within one. A
    pattern without a slash matches the basename at any depth, so `*.py`
    and `.env` behave the way people expect.
    """
    if "/" not in pattern:
        path = path.rsplit("/", 1)[-1]
    return bool(_glob_regex(pattern).match(path))


def _run(argv, cwd, timeout, stdin_data=None, extra_env=None):
    """Run argv with full hygiene; return (raw output bytes, exit code, timed_out)."""
    with tempfile.TemporaryFile() as out:
        # Output goes to a file, not a pipe: a backgrounded grandchild that
        # inherits the pipe would otherwise hold us open until the timeout.
        proc = subprocess.Popen(
            argv, cwd=cwd, env=scrubbed_env(extra_env),
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
        )
        if stdin_data is not None:
            proc.stdin.write(stdin_data)
            proc.stdin.close()
        try:
            code, timed_out = proc.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            code, timed_out = None, True
        return _read_capped(out), code, timed_out


def _read_capped(f) -> bytes:
    size = f.seek(0, os.SEEK_END)
    f.seek(0)
    if size <= _READ_ALL_LIMIT:
        return f.read()
    head = f.read(HEAD_KEEP * 4)
    f.seek(-TAIL_KEEP * 4, os.SEEK_END)
    return head + f"\n... [{size - (HEAD_KEEP + TAIL_KEEP) * 4} bytes skipped] ...\n".encode() + f.read()


def run_process(argv, cwd=None, timeout=None, stdin_data=None, extra_env=None) -> ExecResult:
    """Run argv with the standard hygiene and return a capped ExecResult."""
    raw, code, timed_out = _run(argv, cwd, timeout, stdin_data, extra_env)
    text, truncated = cap_output(raw.decode("utf-8", "replace"))
    return ExecResult(text, code, timed_out, truncated)


def _kill_group(proc) -> int | None:
    """SIGTERM the process group, SIGKILL it after 2s, and reap the leader."""
    for sig, grace in ((signal.SIGTERM, 2), (signal.SIGKILL, None)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            break
        try:
            return proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            continue
    return proc.wait()


def tail_text(path: str, tail_chars: int) -> str:
    """Return roughly the last `tail_chars` characters of a log file."""
    with open(path, "rb") as f:
        size = f.seek(0, os.SEEK_END)
        f.seek(max(0, size - tail_chars * 4))
        return f.read().decode("utf-8", "replace")[-tail_chars:]


class JobTable:
    """Background jobs run as local processes, each in its own process group."""

    def __init__(self, log_dir: str):
        """Jobs log to `<log_dir>/<id>.log`."""
        self.log_dir = log_dir
        self.jobs = {}  # id -> {"proc", "command", "log"}
        self._count = 0
        atexit.register(self.stop_all)

    def new_id(self) -> str:
        """Reserve the next job id (`job_<n>`)."""
        self._count += 1
        return f"job_{self._count}"

    def start(self, job_id, argv, cwd, command, extra_env=None) -> str:
        """Start argv in the background, logging stdout+stderr to the job's log."""
        os.makedirs(self.log_dir, exist_ok=True)
        log = os.path.join(self.log_dir, f"{job_id}.log")
        with open(log, "wb") as f:
            proc = subprocess.Popen(argv, cwd=cwd, env=scrubbed_env(extra_env),
                                    stdin=subprocess.DEVNULL, stdout=f,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        self.jobs[job_id] = {"proc": proc, "command": command, "log": log}
        return job_id

    def get(self, job_id):
        """Return the job record or raise KeyError naming the known ids."""
        if job_id not in self.jobs:
            known = ", ".join(self.jobs) or "none"
            raise KeyError(f"no job {job_id!r}; known jobs: {known}")
        return self.jobs[job_id]

    def output(self, job_id, tail_chars) -> tuple[str, bool]:
        """Return (log tail, still_running)."""
        job = self.get(job_id)
        return tail_text(job["log"], tail_chars), job["proc"].poll() is None

    def stop(self, job_id) -> str:
        """Stop a job's whole process group."""
        job = self.get(job_id)
        if job["proc"].poll() is not None:
            return f"{job_id} had already exited with code {job['proc'].returncode}"
        code = _kill_group(job["proc"])
        return f"stopped {job_id} (exit {code})"

    def list(self) -> list[tuple[str, str, bool]]:
        """Return (id, command, running) for every job."""
        return [(jid, j["command"], j["proc"].poll() is None) for jid, j in self.jobs.items()]

    def stop_all(self) -> None:
        """Stop every job that is still running."""
        for job in self.jobs.values():
            if job["proc"].poll() is None:
                _kill_group(job["proc"])


class Environment(ABC):
    """Where tools run: file operations, commands, and background jobs."""

    name: str = ""          # "local" | "sandbox" | "docker" | "ssh"
    isolation: str = ""     # "none" | "os" | "container" | "remote"
    root: str = ""          # absolute workspace root as the environment sees it
    exec_network: bool = False

    def resolve(self, path: str) -> str:
        """Canonicalize `path` and return it, or raise if it escapes the root."""
        canonical = self._canonical(path)
        if canonical != self.root and not canonical.startswith(self.root.rstrip("/") + "/"):
            raise PermissionError(f"{path!r} resolves outside the workspace ({self.root})")
        return canonical

    def display(self, path: str) -> str:
        """Show a resolved path relative to the root, for messages to the model."""
        if path == self.root:
            return "."
        return path[len(self.root.rstrip("/")) + 1:] if path.startswith(self.root) else path

    @abstractmethod
    def _canonical(self, path: str) -> str:
        """Absolute path with symlinks and `..` flattened, as the env sees it."""

    @abstractmethod
    def read_bytes(self, path: str) -> bytes:
        """Read a file inside the workspace."""

    @abstractmethod
    def write_bytes(self, path: str, data: bytes) -> None:
        """Write a file inside the workspace, creating parent directories."""

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Whether the path exists (inside the workspace)."""

    @abstractmethod
    def is_dir(self, path: str) -> bool:
        """Whether the path is a directory (inside the workspace)."""

    @abstractmethod
    def list_files(self, limit: int) -> list[str]:
        """Relative file paths, ignore set applied, sorted, at most `limit`."""

    @abstractmethod
    def search(self, regex: str, glob: str, limit: int, exclude=None) -> list[tuple[str, int, str]]:
        """(relative path, line number, line) for lines matching regex.

        `exclude(relpath) -> bool` drops files before they are read.
        """

    @abstractmethod
    def exec(self, command: str, timeout: float) -> ExecResult:
        """Run a shell command in the workspace with process hygiene."""

    @abstractmethod
    def spawn(self, command: str) -> str:
        """Start a background job and return its id."""

    @abstractmethod
    def job_output(self, job_id: str, tail_chars: int) -> tuple[str, bool]:
        """Return (tail of the job's output, still_running)."""

    @abstractmethod
    def job_stop(self, job_id: str) -> str:
        """Stop a background job and its children."""

    @abstractmethod
    def list_jobs(self) -> list[tuple[str, str, bool]]:
        """Return (id, command, running) for every job started here."""

    @abstractmethod
    def describe(self) -> str:
        """One line for the system prompt: where commands run and what bounds them."""

    @abstractmethod
    def close(self) -> None:
        """Stop jobs and release resources (containers, connections)."""

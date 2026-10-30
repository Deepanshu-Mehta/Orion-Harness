"""Day 2 — SSHEnv: every operation runs on a remote host over SSH.

Concept: the same operations, executed somewhere else. One multiplexed SSH
connection carries file reads, writes, listings, searches and commands.

Design rules:
- Every interpolated value is `shlex.quote`d; the remote shell sees data,
  never syntax.
- The jail check runs on the remote `realpath -m`, so remote symlinks are
  judged where they live.
- Remote commands get the same hygiene as local ones: `env -i` with an
  allowlist, stdin from /dev/null, `timeout` with a kill grace.
- A remote host is not automatically safe (it might be production), so it
  counts as isolated only when constructed with `disposable=True`.
- Limit: unlike a local exec, a backgrounded remote process that keeps the
  channel open holds `exec` until the timeout; use background jobs.
"""

import posixpath
import shlex

from orion.env.base import IGNORE_DIRS, Environment, ExecResult, _run, glob_match, run_process

_SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ControlMaster=auto",
             "-o", "ControlPath=~/.ssh/orion-%C", "-o", "ControlPersist=120",
             "-o", "LogLevel=ERROR"]
_JOB_DIR = "/tmp/orion-jobs"
_REMOTE_ENV = ('env -i PATH="$PATH" HOME="$HOME" USER="$USER" LANG="${LANG:-C.UTF-8}" '
               'TERM=dumb PYTHONUNBUFFERED=1 CI=1')
_STATUS = "__ORION_JOB_STATUS__"


class SSHEnv(Environment):
    """Run on `host`, in the absolute remote directory `root`."""

    name = "ssh"
    isolation = "remote"

    def __init__(self, host: str, root: str, disposable: bool = False, exec_network: bool = True):
        """Target `host` (anything ssh accepts) with workspace `root`."""
        if not root.startswith("/"):
            raise ValueError(f"ssh root must be absolute, got {root!r}")
        self.host = host
        self.root = posixpath.normpath(root)
        self.disposable = disposable
        self.exec_network = exec_network
        self._jobs = {}  # id -> {"pid", "command"}
        self._count = 0

    def _argv(self, remote_cmd: str) -> list[str]:
        return ["ssh", *_SSH_OPTS, "--", self.host, remote_cmd]

    def _ssh(self, remote_cmd: str, timeout: float = 60, stdin_data=None) -> ExecResult:
        return run_process(self._argv(remote_cmd), timeout=timeout, stdin_data=stdin_data)

    def _check(self, remote_cmd: str, timeout: float = 60, stdin_data=None) -> str:
        result = self._ssh(remote_cmd, timeout, stdin_data)
        if result.exit_code != 0:
            raise OSError(f"remote command failed on {self.host}: {result.output.strip()[:400]}")
        return result.output

    def _canonical(self, path: str) -> str:
        if not path.startswith("/"):
            path = posixpath.join(self.root, path)
        return self._check(f"realpath -m -- {shlex.quote(path)}").strip()

    def read_bytes(self, path: str) -> bytes:
        """Read a remote file as raw bytes."""
        raw, code, _ = _run(self._argv(f"cat -- {shlex.quote(self.resolve(path))}"),
                            None, 120)
        if code != 0:
            raise OSError(raw.decode("utf-8", "replace").strip()[:400] or f"cannot read {path}")
        return raw

    def write_bytes(self, path: str, data: bytes) -> None:
        """Write a remote file from stdin, creating parent directories."""
        target = self.resolve(path)
        parent = shlex.quote(posixpath.dirname(target))
        self._check(f"mkdir -p -- {parent} && cat > {shlex.quote(target)}", 120, data)

    def exists(self, path: str) -> bool:
        """Whether the remote path exists."""
        return self._ssh(f"test -e {shlex.quote(self.resolve(path))}").exit_code == 0

    def is_dir(self, path: str) -> bool:
        """Whether the remote path is a directory."""
        return self._ssh(f"test -d {shlex.quote(self.resolve(path))}").exit_code == 0

    def list_dir(self, path: str) -> list[str]:
        """Names directly inside a remote directory, sorted."""
        target = shlex.quote(self.resolve(path))
        out = self._ssh(f"test -d {target} && ls -1A -- {target}").output
        return sorted(line for line in out.splitlines() if line)

    def _prunes(self) -> str:
        names = " -o ".join(f"-name {shlex.quote(d)}" for d in sorted(IGNORE_DIRS))
        return f"\\( {names} \\) -prune -o"

    def list_files(self, limit: int) -> list[str]:
        """Remote relative file paths, ignore set applied, sorted."""
        out = self._check(f"cd {shlex.quote(self.root)} && find . {self._prunes()} "
                          f"-type f -print | sort | head -n {int(limit)}", 120)
        return [line[2:] for line in out.splitlines() if line.startswith("./")]

    def search(self, regex: str, glob: str, limit: int, exclude=None) -> list[tuple[str, int, str]]:
        """grep -rnIE on the remote host; glob and exclude are applied here."""
        excludes = " ".join(f"--exclude-dir={shlex.quote(d)}" for d in sorted(IGNORE_DIRS))
        # Over-fetch, since the glob and exclude filters run after the cut.
        out = self._ssh(f"cd {shlex.quote(self.root)} && grep -rnIE {excludes} "
                        f"-e {shlex.quote(regex)} . | head -n {int(limit) * 5 + 200}", 120).output
        hits = []
        for line in out.splitlines():
            path, _, rest = line.partition(":")
            lineno, _, text = rest.partition(":")
            rel = path[2:] if path.startswith("./") else path
            if not lineno.isdigit() or not glob_match(rel, glob) or (exclude and exclude(rel)):
                continue
            hits.append((rel, int(lineno), text))
            if len(hits) >= limit:
                break
        return hits

    def exec(self, command: str, timeout: float) -> ExecResult:
        """Run a command remotely under env -i and timeout."""
        remote = (f"cd {shlex.quote(self.root)} && {_REMOTE_ENV} timeout -k 2 "
                  f"{int(timeout + 0.999)} /bin/sh -c {shlex.quote(command)} < /dev/null")
        result = self._ssh(remote, timeout=timeout + 10)
        if result.exit_code == 124:  # GNU timeout's code for "timed out"
            return ExecResult(result.output, None, True, result.truncated)
        return result

    def spawn(self, command: str) -> str:
        """Start a remote background job in its own session."""
        self._count += 1
        job_id = f"job_{self._count}"
        log = f"{_JOB_DIR}/{job_id}.log"
        remote = (f"mkdir -p {_JOB_DIR} && cd {shlex.quote(self.root)} && "
                  f"nohup {_REMOTE_ENV} setsid /bin/sh -c {shlex.quote(command)} "
                  f"> {log} 2>&1 < /dev/null & echo $!")
        pid = self._check(remote).strip().splitlines()[-1]
        if not pid.isdigit():
            raise OSError(f"could not start job on {self.host}: {pid[:200]}")
        self._jobs[job_id] = {"pid": pid, "command": command}
        return job_id

    def _job(self, job_id):
        if job_id not in self._jobs:
            raise KeyError(f"no job {job_id!r}; known jobs: {', '.join(self._jobs) or 'none'}")
        return self._jobs[job_id]

    def _running(self, pid: str) -> bool:
        return self._ssh(f"kill -0 {pid} 2>/dev/null").exit_code == 0

    def job_output(self, job_id: str, tail_chars: int) -> tuple[str, bool]:
        """Return (tail of the remote log, still_running)."""
        pid = self._job(job_id)["pid"]
        out = self._ssh(f"tail -c {int(tail_chars)} -- {_JOB_DIR}/{job_id}.log; "
                        f"printf '\\n{_STATUS}%s' \"$(kill -0 {pid} 2>/dev/null && echo 1)\"").output
        text, _, status = out.rpartition(_STATUS)
        return text.rstrip("\n"), status.strip() == "1"

    def job_stop(self, job_id: str) -> str:
        """Stop the remote job's process group."""
        pid = self._job(job_id)["pid"]
        if not self._running(pid):
            return f"{job_id} had already exited"
        self._ssh(f"kill -TERM -- -{pid} 2>/dev/null; sleep 2; kill -KILL -- -{pid} 2>/dev/null; true")
        return f"stopped {job_id}"

    def list_jobs(self) -> list[tuple[str, str, bool]]:
        """Return (id, command, running) for every remote job."""
        return [(jid, j["command"], self._running(j["pid"])) for jid, j in self._jobs.items()]

    def describe(self) -> str:
        """One line naming the host and whether it is disposable."""
        safety = ("disposable host" if self.disposable
                  else "NOT marked disposable: treat it as a real machine")
        return f"ssh: commands run on {self.host} in {self.root} ({safety})."

    def close(self) -> None:
        """Stop remote jobs and close the shared connection."""
        for job_id, _, running in self.list_jobs():
            if running:
                self.job_stop(job_id)
        run_process(["ssh", *_SSH_OPTS, "-O", "exit", self.host], timeout=10)

"""Day 2 — DockerEnv: commands run in a locked-down, long-lived container.

Concept: a container bounds what a command can reach: its own filesystem,
no network by default, no capabilities, a pid and memory limit. The
workspace is bind-mounted at /workspace, so file tools use the local mount
(fast) while the model sees container paths.

Design rules:
- One container per environment, started on first use, removed on close.
- `.orion/` is mounted read-only over the workspace: memory, skills and
  policy reload next session, so a command must not be able to rewrite them.
- The container runs as the invoking uid:gid, so files it writes in the
  workspace stay owned by the user.
- Killing the local `docker exec` client does not kill the process inside
  the container, so timeouts run `timeout` in the container and jobs record
  an in-container process group that `job_stop` signals.
"""

import math
import os
import shutil
import subprocess
import uuid

from orion.env.base import run_process
from orion.env.local import LocalEnv

# The docker CLI itself needs these to find the daemon; they are not
# forwarded into the container.
_DOCKER_VARS = ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "DOCKER_CERT_PATH",
                "DOCKER_TLS_VERIFY")
_JOB_DIR = "/tmp/orion-jobs"


class DockerEnv(LocalEnv):
    """Run commands inside a container with the workspace mounted at /workspace."""

    name = "docker"
    isolation = "container"

    def __init__(self, root: str, image: str = "python:3.12-slim", exec_network: bool = False):
        """Prepare (but do not yet start) a container for `root`."""
        if shutil.which("docker") is None:
            raise RuntimeError("docker is not installed or not on PATH")
        super().__init__(root)
        self.root = "/workspace"
        self.image = image
        self.exec_network = exec_network
        self.container = None

    @staticmethod
    def available() -> bool:
        """Whether a docker daemon answers."""
        if shutil.which("docker") is None:
            return False
        return subprocess.run(["docker", "info"], capture_output=True).returncode == 0

    def _extra_env(self) -> dict:
        return {k: os.environ[k] for k in _DOCKER_VARS if k in os.environ}

    def _docker(self, *args, timeout=120):
        result = run_process(["docker", *args], timeout=timeout, extra_env=self._extra_env())
        if result.exit_code != 0:
            raise RuntimeError(f"docker {args[0]} failed: {result.output.strip()[:400]}")
        return result.output.strip()

    def _ensure(self) -> str:
        if self.container is None:
            name = f"orion-{uuid.uuid4().hex[:10]}"
            os.makedirs(os.path.join(self.host_root, ".orion"), exist_ok=True)
            args = ["run", "-d", "--name", name]
            if not self.exec_network:
                args += ["--network", "none"]
            args += ["--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                     "--pids-limit", "512", "--memory", "2g",
                     "--user", f"{os.getuid()}:{os.getgid()}",
                     "-v", f"{self.host_root}:/workspace",
                     "-v", f"{self.host_root}/.orion:/workspace/.orion:ro",
                     "-w", "/workspace",
                     self.image, "sleep", "infinity"]
            self._docker(*args, timeout=600)  # may pull the image
            self.container = name
        return self.container

    def _wrap(self, command: str, timeout: float | None) -> list[str]:
        argv = ["docker", "exec", "-i", "-w", "/workspace", self._ensure()]
        if timeout is not None:
            argv += ["timeout", "-k", "2", str(math.ceil(timeout))]
        return argv + ["/bin/sh", "-c", command]

    def _wrap_job(self, command: str, job_id: str) -> list[str]:
        # setsid makes the inner shell a group leader; its pid is saved so
        # job_stop can signal the whole in-container group.
        inner = f'mkdir -p {_JOB_DIR}; echo $$ > {_JOB_DIR}/{job_id}.pid; exec /bin/sh -c "$0"'
        return ["docker", "exec", "-i", "-w", "/workspace", self._ensure(),
                "setsid", "/bin/sh", "-c", inner, command]

    def job_stop(self, job_id: str) -> str:
        """Signal the job's process group inside the container, then the client."""
        self._jobs.get(job_id)
        if self.container is not None:
            run_process(["docker", "exec", self.container, "/bin/sh", "-c",
                         f"kill -TERM -- -$(cat {_JOB_DIR}/{job_id}.pid) 2>/dev/null"],
                        timeout=10, extra_env=self._extra_env())
        return super().job_stop(job_id)

    def describe(self) -> str:
        """One line naming the container and its limits."""
        net = "network ON" if self.exec_network else "network OFF"
        return (f"docker ({self.image}): commands run in a container with the workspace at "
                f"/workspace; no capabilities, {net}, 2 GB memory, 512 pids.")

    def close(self) -> None:
        """Stop jobs and remove the container."""
        if self.container is not None:
            for job_id, _, running in self.list_jobs():
                if running:
                    self.job_stop(job_id)
            run_process(["docker", "rm", "-f", self.container], timeout=60,
                        extra_env=self._extra_env())
            self.container = None
        super().close()

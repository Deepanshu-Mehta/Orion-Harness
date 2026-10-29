"""Day 2 — environment factory.

Concept: one string picks where the agent works: "local", "sandbox",
"docker", "docker:<image>", or "ssh:<host>:<abs_root>".

Design rules: options a backend does not take are dropped, not passed
through, so one CLI can offer every flag.
"""

from orion.env.base import IGNORE_DIRS, Environment, ExecResult, glob_match
from orion.env.docker import DockerEnv
from orion.env.local import LocalEnv
from orion.env.sandbox import SandboxEnv
from orion.env.ssh import SSHEnv

__all__ = ["DockerEnv", "Environment", "ExecResult", "IGNORE_DIRS", "LocalEnv",
           "SSHEnv", "SandboxEnv", "glob_match", "make_env"]


def _pick(opts: dict, *names) -> dict:
    return {k: opts[k] for k in names if opts.get(k) is not None}


def make_env(spec: str, root: str, **opts) -> Environment:
    """Build an Environment from a spec string; `root` is ignored for ssh."""
    kind, _, rest = spec.partition(":")
    if kind == "local" and not rest:
        return LocalEnv(root)
    if kind == "sandbox" and not rest:
        return SandboxEnv(root, **_pick(opts, "exec_network", "writable"))
    if kind == "docker":
        return DockerEnv(root, image=rest or "python:3.12-slim", **_pick(opts, "exec_network"))
    if kind == "ssh":
        host, _, remote_root = rest.partition(":")
        if not host or not remote_root:
            raise ValueError("ssh spec must be ssh:<host>:<absolute root>")
        return SSHEnv(host, remote_root, **_pick(opts, "disposable", "exec_network"))
    raise ValueError(f"unknown environment {spec!r}; use local, sandbox, docker[:image], "
                     "or ssh:<host>:<abs_root>")

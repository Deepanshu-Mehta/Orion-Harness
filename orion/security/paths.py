"""Day 2 — protected paths: files the agent never reads, never writes, or asks about.

Concept: some files are dangerous regardless of mode. Secrets must never
enter the context (whatever enters the context reaches the provider). The
agent's own policy, logs and git internals must never be rewritten by the
agent. CI and dependency manifests are supply-chain surfaces that deserve a
human look.

Design rules:
- Patterns without a leading `~` or `/` match the path relative to the
  workspace root; `~/...` patterns match the resolved absolute path.
- A `!pattern` entry is an exception (`.env.example` is not a secret).
  Exceptions exist only in the defaults; policy files cannot add them,
  because an exception is a way to loosen a deny.
"""

import posixpath

from orion.env.base import glob_match

DEFAULT_DENY = [".env", ".env.*", "!.env.example", "**/*.pem", "**/*.key", "**/id_rsa*",
                "**/id_ed25519*", "~/.ssh/**", "~/.aws/**", "~/.netrc", "~/.orion/**"]
DEFAULT_DENY_WRITE = [".git/**", ".orion/**"]
DEFAULT_ASK_WRITE = [".github/workflows/**", "**/Dockerfile", "Makefile", "package.json",
                     "pyproject.toml", "requirements*.txt", "package-lock.json", "yarn.lock",
                     "pnpm-lock.yaml", "poetry.lock", "uv.lock", "Pipfile.lock",
                     "Cargo.lock", "Gemfile.lock", "go.sum"]


def _target(abs_path: str, pattern: str, root: str, home: str):
    """(path to test, pattern to test it with), or None if they cannot match."""
    if pattern.startswith("~"):
        return abs_path, home + pattern[1:]
    if pattern.startswith("/"):
        return abs_path, pattern
    if abs_path == root:
        return None
    prefix = root.rstrip("/") + "/"
    if not abs_path.startswith(prefix):
        return None
    return abs_path[len(prefix):], pattern


def match(abs_path: str, patterns: list[str], root: str, home: str) -> str | None:
    """Return the first pattern that protects `abs_path`, honoring `!` exceptions."""
    abs_path = posixpath.normpath(abs_path)
    for pattern in patterns:
        if pattern.startswith("!"):
            continue
        target = _target(abs_path, pattern, root, home)
        if target is None or not glob_match(*target):
            continue
        excepted = any((t := _target(abs_path, e[1:], root, home)) and glob_match(*t)
                       for e in patterns if e.startswith("!"))
        if not excepted:
            return pattern
    return None

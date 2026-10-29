"""Day 2 — SandboxEnv: the same machine, with every command in an OS jail.

Concept: pattern rules can be talked around; the kernel cannot. File tools
run in the trusted harness process (already jailed by `resolve`), but every
exec and spawn is wrapped in seatbelt (`sandbox-exec`, macOS) or bubblewrap
(`bwrap`, Linux), so a command that slips past the gate still cannot write
outside the workspace, read credentials, or reach the network.

Design rules:
- Never fall back silently: if the backend is missing, construction fails.
- Paths in the profile are realpaths (`/tmp` is `/private/tmp` on macOS).
- `.orion/` and `.git/hooks` stay read-only even inside the workspace, so a
  command cannot rewrite the agent's policy or plant a hook that runs later
  outside the sandbox.
- Bounds the filesystem and network only, not CPU, memory or time.
"""

import os
import shutil
import sys

from orion.env.local import LocalEnv


def _sb_str(path: str) -> str:
    """Quote a path as a seatbelt string literal."""
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _sb_regex(prefix: str) -> str:
    """A seatbelt regex literal matching paths that start with `prefix`."""
    escaped = "".join("\\" + c if c in ".^$*+?()[]{}|\\" else c for c in prefix)
    return '#"^' + escaped.replace('"', '\\"') + '"'


class SandboxEnv(LocalEnv):
    """LocalEnv whose commands run under seatbelt (macOS) or bubblewrap (Linux)."""

    name = "sandbox"
    isolation = "os"

    def __init__(self, root: str, exec_network: bool = False, writable: bool = True):
        """Jail commands to `root`; network and writes per the flags."""
        if not self.available():
            raise RuntimeError(
                "sandbox backend unavailable: needs sandbox-exec (macOS) or bwrap (Linux). "
                "Install it, or choose --env docker or --env local explicitly.")
        super().__init__(root)
        self.exec_network = exec_network
        self.writable = writable
        # bwrap needs the mount point to exist, and the jobs log dir lives here.
        os.makedirs(os.path.join(self.host_root, ".orion"), exist_ok=True)
        self.home = os.path.realpath(os.path.expanduser("~"))
        if sys.platform == "darwin":
            self._profile = self._seatbelt_profile()

    @staticmethod
    def available() -> bool:
        """Whether this platform's sandbox backend is installed."""
        if sys.platform == "darwin":
            return shutil.which("sandbox-exec") is not None
        if sys.platform.startswith("linux"):
            return shutil.which("bwrap") is not None
        return False

    def _seatbelt_profile(self) -> str:
        root, home = self.host_root, self.home
        tmp_dirs = ['(subpath "/private/tmp")', '(subpath "/private/var/folders")']
        lines = ["(version 1)", "(allow default)", "(deny file-write*)"]
        writable = [] if not self.writable else [f"(subpath {_sb_str(root)})"]
        lines.append("(allow file-write* " + " ".join(
            writable + tmp_dirs + ['(literal "/dev/null")', '(regex #"^/dev/tty")']) + ")")
        # Later rules win in seatbelt, so these denies override the allows
        # above, including when the workspace itself lives under a temp dir.
        protected = [root] if not self.writable else [root + "/.orion", root + "/.git/hooks"]
        lines.append("(deny file-write* " + " ".join(f"(subpath {_sb_str(p)})" for p in protected) + ")")
        secrets = [f"(subpath {_sb_str(os.path.join(home, d))})"
                   for d in (".ssh", ".aws", ".config/gcloud", ".orion")]
        secrets.append(f"(literal {_sb_str(os.path.join(home, '.netrc'))})")
        secrets.append(f"(regex {_sb_regex(root + '/.env')})")
        lines.append("(deny file-read* " + " ".join(secrets) + ")")
        if not self.exec_network:
            lines.append("(deny network*)")
            lines.append("(allow network* (local unix))")
        return "\n".join(lines)

    def _bwrap_argv(self) -> list[str]:
        root, home = self.host_root, self.home
        argv = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
                "--tmpfs", "/tmp",
                "--bind" if self.writable else "--ro-bind", root, root,
                "--ro-bind", f"{root}/.orion", f"{root}/.orion"]
        if os.path.isdir(f"{root}/.git/hooks"):
            argv += ["--ro-bind", f"{root}/.git/hooks", f"{root}/.git/hooks"]
        for secret in (".ssh", ".aws", ".config/gcloud", ".orion"):
            if os.path.isdir(os.path.join(home, secret)):
                argv += ["--tmpfs", os.path.join(home, secret)]
        argv += ["--unshare-pid", "--die-with-parent", "--new-session"]
        if not self.exec_network:
            argv.append("--unshare-net")
        return argv + ["--chdir", root, "--"]

    def _wrap(self, command: str, timeout: float | None) -> list[str]:
        if sys.platform == "darwin":
            return ["sandbox-exec", "-p", self._profile, "/bin/sh", "-c", command]
        return self._bwrap_argv() + ["/bin/sh", "-c", command]

    def describe(self) -> str:
        """One line naming the jail and what it does and does not bound."""
        backend = "macOS seatbelt" if sys.platform == "darwin" else "bubblewrap"
        writes = (f"writes only inside {self.root} and temp dirs" if self.writable
                  else "NO writes except temp dirs")
        net = "network ON" if self.exec_network else "network OFF"
        return (f"sandbox ({backend}): commands run in {self.root}; {writes}; no access to "
                f"~/.ssh, ~/.aws, or .env files; {net}. CPU and memory are not limited.")

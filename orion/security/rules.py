"""Day 2 — command rules: hard denies, the ask list, and the read-only allowlist.

Concept: the outer ring of the gate. A shell command is split into the
segments the shell would run (including `$(...)`, backticks, and `sh -c`
strings), and each segment is judged by its program and arguments, not by
substring search, so `grep -r "rm -rf /" docs/` is text while `rm -rf /`
is a command.

Design rules:
- Hard denies apply in every mode and every environment; no approval or
  session rule can lift them.
- Pattern rules are known to leak (variables, aliases, scripts written then
  run); they catch accidents and obvious mistakes. OS isolation is the
  boundary we rely on. docs/security.md lists the known gaps.
- Read-only means "cannot change or exfiltrate anything": no redirection,
  no substitution, no variable expansion, every segment on the allowlist.
"""

import os
import posixpath
import re
import shlex
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    """A named hard-deny rule."""

    name: str
    reason: str


@dataclass
class Segment:
    """One simple command the shell would run."""

    raw: str
    op: str             # the operator that follows it: ";", "&&", "||", "|", "&", ""
    tokens: list        # shlex tokens, grouping punctuation removed
    prog: str = ""      # basename of the program after wrappers/assignments
    args: list = None   # arguments after the program


_OPERATORS = ("&&", "||", "|&", ";", "|", "&", "\n")
_WRAPPERS = {"env", "nohup", "time", "command", "exec", "nice", "builtin", "stdbuf",
             "timeout", "xargs", "caffeinate"}
_SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "fish"}
INTERPRETERS = _SHELLS | {"python", "python3", "perl", "node", "ruby", "php"}
_HOME_FORMS = ("~", "$HOME", "${HOME}")


def _balanced(s: str, start: int) -> int:
    """Index just past the `)` that closes a group opened before `start`."""
    depth, i, quote = 1, start, None
    while i < len(s):
        c = s[i]
        if quote:
            if c == quote:
                quote = None
            elif c == "\\" and quote == '"':
                i += 1
        elif c in "'\"":
            quote = c
        elif c == "\\":
            i += 1
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(s)


def _split_raw(command: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Split at unquoted operators; also return substitution bodies to judge."""
    parts, subs, buf, quote, i = [], [], [], None, 0
    n = len(command)
    while i < n:
        c = command[i]
        if quote == "'":
            buf.append(c)
            quote = None if c == "'" else quote
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            buf.append(command[i:i + 2])
            i += 2
            continue
        # Substitutions execute even inside double quotes.
        if command.startswith("$(", i) or (not quote and command.startswith(("<(", ">("), i)):
            end = _balanced(command, i + 2)
            subs.append(command[i + 2:end - 1])
            buf.append(command[i:end])
            i = end
            continue
        if c == "`":
            end = command.find("`", i + 1)
            end = n if end == -1 else end
            subs.append(command[i + 1:end])
            buf.append(command[i:end + 1])
            i = end + 1
            continue
        if quote == '"':
            buf.append(c)
            quote = None if c == '"' else quote
            i += 1
            continue
        if c in "'\"":
            quote = c
            buf.append(c)
            i += 1
            continue
        if c == "#" and (not buf or buf[-1][-1:].isspace()):
            end = command.find("\n", i)
            i = n if end == -1 else end
            continue
        op = next((o for o in _OPERATORS if command.startswith(o, i)), None)
        # `2>&1` and `&>file` are redirections, not the background operator.
        if op == "&" and ((buf and buf[-1][-1:] in "<>") or command.startswith("&>", i)):
            op = None
        if op:
            parts.append(("".join(buf).strip(), op.strip() or ";"))
            buf = []
            i += len(op)
            continue
        buf.append(c)
        i += 1
    parts.append(("".join(buf).strip(), ""))
    return [(raw, op) for raw, op in parts if raw], subs


def _tokens(raw: str) -> list[str]:
    try:
        tokens = shlex.split(raw, comments=True)
    except ValueError:
        tokens = re.findall(r"[^\s'\"]+", raw)  # unbalanced quotes: best effort
    tokens = [t for t in tokens if t not in ("(", ")", "{", "}", "!")]
    if tokens and tokens[0].startswith("("):
        tokens[0] = tokens[0].lstrip("(")
    if tokens and tokens[-1].endswith(")") and "(" not in tokens[-1]:
        tokens[-1] = tokens[-1].rstrip(")")
    return [t for t in tokens if t]


def _program(tokens: list[str]) -> tuple[str, list[str]]:
    """Skip assignments and wrapper commands; return (program, args)."""
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t):
            i += 1
            continue
        base = posixpath.basename(t)
        if base in _WRAPPERS:
            i += 1
            # Skip the wrapper's own flags and arguments (e.g. `timeout 10`, `nice -n 5`).
            while i < len(tokens) and (tokens[i].startswith("-") or tokens[i][:1].isdigit()
                                       or re.match(r"^[A-Za-z_]\w*=", tokens[i])):
                i += 1
            continue
        return base, tokens[i + 1:]
    return "", []


def split_command(command: str) -> list[Segment]:
    """Every simple command `command` would run, including nested ones."""
    segments, pending = [], [command]
    while pending:
        parts, subs = _split_raw(pending.pop(0))
        for raw, op in parts:
            tokens = _tokens(raw)
            prog, args = _program(tokens)
            segments.append(Segment(raw, op, tokens, prog, args))
            # `sh -c "..."` and `eval "..."` run their string as a command.
            if prog in _SHELLS and "-c" in args and args.index("-c") + 1 < len(args):
                pending.append(args[args.index("-c") + 1])
            elif prog == "eval" and args:
                pending.append(" ".join(args))
        pending.extend(subs)
    return segments


def _expand_home(token: str, home: str) -> str:
    for form in _HOME_FORMS:
        if token == form or token.startswith(form + "/"):
            return home + token[len(form):]
    return token


def _outside(token: str, root: str, home: str, cwd_unsafe: bool) -> bool:
    """Whether a path argument reaches `/`, home, or anywhere outside root."""
    core = re.sub(r"(/\.?\*)+$", "", token.strip())
    if not core and token.startswith("/"):
        core = "/"
    path = _expand_home(core, home) if core else ""
    if path.startswith("/"):
        norm = posixpath.normpath(path)
        # / and home are never fair game, even when the workspace is home itself.
        if norm in ("/", home):
            return True
        return not (norm == root or norm.startswith(root.rstrip("/") + "/"))
    if "$" in path or "`" in path:
        return False  # unknown expansion: left to the ask list and the sandbox
    if cwd_unsafe:
        return True
    norm = posixpath.normpath(posixpath.join(root, path or "."))
    return not (norm == root or norm.startswith(root.rstrip("/") + "/"))


def _recursive(args: list[str]) -> bool:
    return any(a in ("--recursive",) or (re.match(r"^-[A-Za-z]+$", a) and ("r" in a or "R" in a))
               for a in args)


def _operands(args: list[str]) -> list[str]:
    if "--" in args:
        i = args.index("--")
        return [a for a in args[:i] if not a.startswith("-")] + args[i + 1:]
    return [a for a in args if not a.startswith("-")]


def _force_push(seg: Segment) -> bool:
    if seg.prog != "git" or "push" not in seg.args:
        return False
    after = seg.args[seg.args.index("push") + 1:]
    return any(a in ("--force", "--force-if-includes") or a.startswith("--force-with-lease")
               or re.match(r"^-[A-Za-z]*f[A-Za-z]*$", a) or a.startswith("+") for a in after)


def _find_delete(seg: Segment, ctx) -> bool:
    if seg.prog != "find":
        return False
    starts = []
    for a in seg.args:
        if a.startswith("-") or a in ("(", "!"):
            break
        starts.append(a)
    deletes = "-delete" in seg.args or any(
        a in ("-exec", "-execdir", "-ok", "-okdir") and i + 1 < len(seg.args)
        and posixpath.basename(seg.args[i + 1]) == "rm"
        for i, a in enumerate(seg.args))
    return deletes and any(_outside(s, *ctx) for s in starts or ["."])


SEGMENT_RULES = [
    (Rule("rm-outside-workspace", "recursive delete of /, the home directory, or a path "
          "outside the workspace"),
     lambda s, ctx: s.prog == "rm" and _recursive(s.args)
     and any(_outside(t, *ctx) for t in _operands(s.args))),
    (Rule("privilege-escalation", "commands must not run as another user or as root"),
     lambda s, ctx: s.prog in ("sudo", "doas", "su")),
    (Rule("disk-format", "formatting or raw-writing a disk destroys data"),
     lambda s, ctx: s.prog.startswith("mkfs")
     or (s.prog == "dd" and any(a.startswith("of=/dev/") for a in s.args))),
    (Rule("force-push", "force-pushing rewrites shared history"), lambda s, ctx: _force_push(s)),
    (Rule("recursive-permissions-outside", "recursive ownership or permission change on /, "
          "home, or outside the workspace"),
     lambda s, ctx: s.prog in ("chmod", "chown", "chgrp") and _recursive(s.args)
     and any(_outside(t, *ctx) for t in _operands(s.args)[1:])),
    (Rule("power", "shutting down or rebooting the machine"),
     lambda s, ctx: s.prog in ("shutdown", "reboot", "halt", "poweroff")
     or (s.prog == "systemctl" and any(a in ("poweroff", "reboot", "halt") for a in s.args))),
    (Rule("find-delete-outside", "bulk delete with find rooted at /, home, or outside the "
          "workspace"), _find_delete),
]

_FORK_BOMB = re.compile(r"(\S+)\s*\(\)\s*\{[^}]*\1\s*\|\s*\1\s*&")
_DEVICE_WRITE = re.compile(r">\s*/dev/(r?disk|sd[a-z]|nvme|hd[a-z]|mmcblk)")
_PIPE_TO_INTERPRETER = Rule("pipe-to-interpreter", "running code downloaded from the network "
                            "without reviewing it")


def _blank_quotes(raw: str) -> str:
    return re.sub(r"'[^']*'|\"(?:\\.|[^\"\\])*\"", "''", raw)


def hard_deny(command: str, root: str, home: str, extra_patterns=()) -> Rule | None:
    """Return the first hard-deny rule `command` breaks, or None."""
    if _FORK_BOMB.search(_blank_quotes(command)):
        return Rule("fork-bomb", "a fork bomb exhausts the machine's process table")
    for pattern in extra_patterns:
        if re.search(pattern, command):
            return Rule("policy", f"matches the policy deny pattern {pattern!r}")
    segments = split_command(command)
    cwd_unsafe = False
    for i, seg in enumerate(segments):
        ctx = (root, home, cwd_unsafe)
        for rule, check in SEGMENT_RULES:
            if check(seg, ctx):
                return rule
        if _DEVICE_WRITE.search(_blank_quotes(seg.raw)):
            return Rule("disk-format", "redirecting output onto a raw disk device destroys data")
        if seg.prog in ("curl", "wget") and seg.op == "|":
            for later in segments[i + 1:]:
                if later.prog in INTERPRETERS:
                    return _PIPE_TO_INTERPRETER
                if later.op != "|":
                    break
        if seg.prog in INTERPRETERS and re.search(r"[$<]\(\s*(curl|wget)\b", seg.raw):
            return _PIPE_TO_INTERPRETER
        if seg.prog == "cd":
            target = _operands(seg.args)[0] if _operands(seg.args) else "~"
            # Once the shell has left the workspace, relative paths are unsafe
            # until it cds back to an absolute path inside root.
            back_inside = (target.startswith("/") or target.startswith(_HOME_FORMS)) and \
                not _outside(target, root, home, False)
            cwd_unsafe = _outside(target, *ctx) or (cwd_unsafe and not back_inside)
    return None


_INSTALL = {"pip": {"install"}, "pip3": {"install"}, "npm": {"install", "i", "ci", "add"},
            "pnpm": {"install", "i", "add"}, "yarn": {"install", "add", ""},
            "brew": {"install", "upgrade"}, "apt": {"install"}, "apt-get": {"install"},
            "uv": {"pip", "add"}, "poetry": {"add", "install"}, "gem": {"install"},
            "cargo": {"install"}}


def ask_reason(command: str, root: str, home: str, exec_network: bool,
               extra_patterns=()) -> str | None:
    """Why `command` needs a human look even when the mode would allow it, or None."""
    for pattern in extra_patterns:
        if re.search(pattern, command):
            return f"matches the policy ask pattern {pattern!r}"
    for seg in split_command(command):
        a, sub = seg.args, (seg.args[0] if seg.args else "")
        if seg.prog == "git" and "push" in a:
            return "git push publishes commits to a remote"
        if seg.prog == "git" and "reset" in a and "--hard" in a:
            return "git reset --hard discards uncommitted work"
        if seg.prog == "git" and "clean" in a and any(
                x == "--force" or re.match(r"^-[a-zA-Z]*f", x) for x in a):
            return "git clean -f deletes untracked files"
        if seg.prog == "rm" and _recursive(a):
            return "recursive delete inside the workspace"
        if seg.prog in _INSTALL and (sub in _INSTALL[seg.prog] or (not a and "" in _INSTALL[seg.prog])):
            return f"{seg.prog} {sub} installs packages (supply-chain risk)".replace("  ", " ")
        if seg.prog.startswith("python") and a[:3] in (["-m", "pip", "install"],):
            return "pip install installs packages (supply-chain risk)"
        if seg.prog in ("docker", "kubectl", "terraform"):
            return f"{seg.prog} acts on infrastructure outside the workspace"
        if seg.prog in ("curl", "wget") and exec_network:
            return f"{seg.prog} reaches the network"
    return None


# (token prefix, runs project code). Code-running entries auto-run only in
# isolated environments: on the bare machine, "run the tests" after
# accept-edits would be unapproved arbitrary execution.
READ_ONLY_ALLOWLIST = [
    (["ls"], False), (["cat"], False), (["head"], False), (["tail"], False), (["wc"], False),
    (["pwd"], False), (["echo"], False), (["which"], False), (["file"], False),
    (["stat"], False), (["tree"], False), (["du"], False), (["df"], False),
    (["git", "status"], False), (["git", "diff"], False), (["git", "log"], False),
    (["git", "show"], False), (["git", "branch"], False), (["git", "rev-parse"], False),
    (["node", "--version"], False),
    (["python3", "-m", "unittest"], True), (["python3", "-m", "pytest"], True),
    (["pytest"], True), (["npm", "test"], True), (["npm", "run", "test"], True),
]
_UNSAFE_SYNTAX = (">", "`", "$(", "<(")
_GIT_WRITE_FLAGS = ("--output", "--ext-diff", "--exec")


def is_read_only(command: str, isolated: bool, extra_prefixes=()) -> bool:
    """Whether every segment is an allowlisted, side-effect-free command."""
    if any(s in command for s in _UNSAFE_SYNTAX):
        return False
    allow = [(p, code) for p, code in READ_ONLY_ALLOWLIST if isolated or not code]
    allow += [(shlex.split(p), False) for p in extra_prefixes]
    segments = split_command(command)
    if not segments or any(seg.op == "&" for seg in segments):
        return False
    for seg in segments:
        if not any(seg.tokens[:len(p)] == p for p, _ in allow):
            return False
        # Unknown expansions could name anything, including a protected path.
        if any(re.search(r"[*?\[$]", _expand_home(t, "/h")) for t in seg.tokens):
            return False
        if seg.tokens[0] == "git" and any(t.startswith(_GIT_WRITE_FLAGS) for t in seg.tokens):
            return False
    return True


def path_operands(command: str, root: str, home: str) -> list[str]:
    """Absolute paths a command names as operands, for protected-path checks."""
    found = []
    for seg in split_command(command):
        if seg.prog == "echo":
            continue  # its arguments are text, not paths
        for tok in seg.args or []:
            if tok.startswith("-"):
                if "=" not in tok:
                    continue
                tok = tok.split("=", 1)[1]
            path = _expand_home(tok, home)
            found.append(posixpath.normpath(path if path.startswith("/")
                                            else posixpath.join(root, path)))
    return found


def default_home() -> str:
    """The real home directory, used to judge `~` in commands."""
    return os.path.realpath(os.path.expanduser("~"))

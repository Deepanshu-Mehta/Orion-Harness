"""Day 2 — the core toolset: a small set of verbs over an Environment.

Concept: a tool earns its slot only if it covers a verb the shell lacks,
can be permitted separately from the shell (so read-only mode works without
one), shapes output for the model better than the shell would, or carries
safety the shell cannot. "Names a task bash already does" is not a reason.

Design rules:
- Tools call the Environment only; they never touch `os` or `subprocess`.
- Output is shaped for the model: line numbers, explicit continuation
  hints, caps, and diffs that show what an edit changed.
- `edit_file` is atomic: every edit is checked against the evolving text
  before anything is written.
- `secret(relpath)`, when given, hides matching files from grep so file
  contents the gate would deny never arrive through a search.
"""

import difflib
import re
from typing import Literal, TypedDict

from orion.env.base import glob_match
from orion.tools.base import Tool, ToolError, tool

READ_LINE_CLIP = 2_000
READ_CHAR_CAP = 50_000
GREP_LINE_CLIP = 200
DIFF_LINE_CAP = 80
JOB_TAIL_CHARS = 4_000
_SCAN_LIMIT = 100_000


class Edit(TypedDict):
    """One exact-text replacement."""

    old: str
    new: str


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _occurrences(text: str, needle: str) -> list[int]:
    starts, i = [], text.find(needle)
    while i != -1:
        starts.append(i)
        i = text.find(needle, i + len(needle))
    return starts


def core_tools(env, secret=None) -> dict[str, Tool]:
    """Build the core tools closed over `env`."""

    @tool("Read a text file with line numbers. For long files, page with offset/limit.",
          kind="read", paths={"path": "read"},
          path="File path, relative to the workspace root",
          offset="1-based line number to start from", limit="Maximum lines to return")
    def read_file(path: str, offset: int = 1, limit: int = 2000) -> str:
        resolved = env.resolve(path)
        if env.is_dir(resolved):
            raise ToolError(f"{path} is a directory; use list_files to see what is inside")
        if offset < 1 or limit < 1:
            raise ToolError("offset and limit must be at least 1")
        data = env.read_bytes(resolved)
        if b"\0" in data[:8192]:
            return f"binary file, {len(data)} bytes"
        lines = data.decode("utf-8", "replace").splitlines()
        if not lines:
            return "(empty file)"
        if offset > len(lines):
            raise ToolError(f"offset {offset} is past the end; the file has {len(lines)} lines")
        out, used = [], 0
        end = min(len(lines), offset - 1 + limit)
        for n in range(offset, end + 1):
            line = lines[n - 1]
            if len(line) > READ_LINE_CLIP:
                line = line[:READ_LINE_CLIP] + f" ... [line clipped, {len(line)} chars]"
            entry = f"{n}\t{line}"
            if used + len(entry) > READ_CHAR_CAP and out:
                out.append(f"[stopped at {READ_CHAR_CAP:,} chars; use offset={n} to continue]")
                return "\n".join(out)
            out.append(entry)
            used += len(entry) + 1
        if end < len(lines):
            out.append(f"[lines {offset}-{end} of {len(lines)}; use offset={end + 1} to continue]")
        return "\n".join(out)

    @tool("List files in the workspace (dependency and build directories are skipped).",
          kind="read", glob="Glob such as '**/*.py' or 'src/*'; a pattern without '/' "
          "matches file names at any depth", limit="Maximum paths to return")
    def list_files(glob: str = "**/*", limit: int = 500) -> str:
        matches = [f for f in env.list_files(_SCAN_LIMIT) if glob_match(f, glob)]
        if not matches:
            return f"no files match {glob!r}"
        shown = matches[:limit]
        if len(matches) > limit:
            shown.append(f"... and {len(matches) - limit} more")
        return "\n".join(shown)

    @tool("Search file contents with a regular expression. Returns path:line: text.",
          kind="read", regex="Python regular expression",
          glob="Only search files matching this glob", limit="Maximum matching lines")
    def grep(regex: str, glob: str = "**/*", limit: int = 200) -> str:
        try:
            re.compile(regex)
        except re.error as e:
            raise ToolError(f"invalid regex {regex!r}: {e}; fix the pattern (escape special "
                            "characters like ( [ . with a backslash)")
        hits = env.search(regex, glob, limit + 1, exclude=secret)
        if not hits:
            return f"no matches for {regex!r} in {glob!r}"
        lines = [f"{p}:{n}: {t[:GREP_LINE_CLIP]}" for p, n, t in hits[:limit]]
        if len(hits) > limit:
            lines.append(f"[more than {limit} matches; narrow the regex or glob]")
        return "\n".join(lines)

    @tool("Create or overwrite a file with the given content.", kind="write",
          paths={"path": "write"}, path="File path, relative to the workspace root",
          content="The complete new file content")
    def write_file(path: str, content: str) -> str:
        resolved = env.resolve(path)
        if env.is_dir(resolved):
            raise ToolError(f"{path} is a directory; choose a file path")
        existed = env.exists(resolved)
        env.write_bytes(resolved, content.encode("utf-8"))
        verb = "overwrote" if existed else "created"
        return f"Wrote {len(content)} chars to {env.display(resolved)} ({verb})"

    @tool("Edit a file by exact text replacement. Each edit's `old` must match exactly "
          "once; edits apply in order and all-or-nothing. Read the file first.",
          kind="write", paths={"path": "write"},
          path="File path, relative to the workspace root",
          edits="List of {old, new} replacements; `old` is copied exactly from the file")
    def edit_file(path: str, edits: list[Edit]) -> str:
        resolved = env.resolve(path)
        if not env.exists(resolved):
            raise ToolError(f"{path} does not exist; use write_file to create it")
        if not edits:
            raise ToolError("edits is empty; pass at least one {old, new}")
        original = env.read_bytes(resolved).decode("utf-8", "replace")
        text = original
        for i, edit in enumerate(edits, 1):
            if not edit["old"]:
                raise ToolError(f"edit {i}: old is empty; copy the exact text to replace")
            found = _occurrences(text, edit["old"])
            if not found:
                raise ToolError(f"edit {i}: old text not found; nothing was written. Read the "
                                "file and copy the text exactly, including whitespace")
            if len(found) > 1:
                at = ", ".join(str(_line_of(text, s)) for s in found[:10])
                raise ToolError(f"edit {i}: old text appears {len(found)} times (at lines {at}); "
                                "nothing was written. Include more surrounding lines")
            text = text[:found[0]] + edit["new"] + text[found[0] + len(edit["old"]):]
        env.write_bytes(resolved, text.encode("utf-8"))
        rel = env.display(resolved)
        diff = list(difflib.unified_diff(original.splitlines(), text.splitlines(),
                                         f"a/{rel}", f"b/{rel}", n=2, lineterm=""))
        if len(diff) > DIFF_LINE_CAP:
            diff = diff[:DIFF_LINE_CAP] + [f"... [{len(diff) - DIFF_LINE_CAP} more diff lines]"]
        return f"Applied {len(edits)} edit(s) to {rel}\n" + "\n".join(diff)

    @tool("Run a shell command in the workspace and return its output. For long-running "
          "servers or watchers, use background_job instead.", kind="exec",
          command="Shell command (run with /bin/sh)", timeout="Seconds before it is killed, 1-600")
    def bash(command: str, timeout: int = 120) -> str:
        if not 1 <= timeout <= 600:
            raise ToolError("timeout must be between 1 and 600 seconds")
        r = env.exec(command, timeout)
        if r.timed_out:
            head = f"[timed out after {timeout}s — partial output]"
        else:
            head = f"[exit {r.exit_code}]"
        return f"{head}\n{r.output}" if r.output else f"{head} (no output)"

    @tool("Manage background processes such as dev servers and watchers: start one, read "
          "the tail of its output, stop it, or list jobs.", kind="exec",
          action="start | output | stop | list", command="Shell command, for start",
          job_id="Job id from start, for output and stop")
    def background_job(action: Literal["start", "output", "stop", "list"],
                       command: str = "", job_id: str = "") -> str:
        if action == "start":
            if not command:
                raise ToolError("start needs a command")
            new_id = env.spawn(command)
            return (f"started {new_id}; read its output with "
                    f"background_job(action='output', job_id='{new_id}')")
        if action == "list":
            jobs = env.list_jobs()
            if not jobs:
                return "no background jobs"
            return "\n".join(f"{j} [{'running' if r else 'exited'}] {c}" for j, c, r in jobs)
        if not job_id:
            raise ToolError(f"{action} needs a job_id; use action='list' to see jobs")
        try:
            if action == "stop":
                return env.job_stop(job_id)
            output, running = env.job_output(job_id, JOB_TAIL_CHARS)
        except KeyError as e:
            raise ToolError(e.args[0]) from None
        state = "running" if running else "exited"
        return f"[{job_id} {state}]\n{output}" if output else f"[{job_id} {state}] (no output yet)"

    tools = [read_file, list_files, grep, write_file, edit_file, bash, background_job]
    return {t.name: t for t in tools}

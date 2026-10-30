"""Day 3 — agent memory: notes that survive the process.

Concept: project instructions are written by humans and obeyed; memory is
written by the agent and treated as facts. They carry different trust, so
they live in different files and get different prompt labels.

Design rules:
- `remember` writes through the Environment from the harness side, because
  `.orion/` is deny-write for the file tools and that must stay true.
- Memory reloads every session, which makes it a persistent injection
  channel. Notes are scanned when written, and the file is scanned when
  loaded, since a cloned repository can ship its own `.orion/memory.md`.
- The file is capped; at the cap the human prunes it. Silent eviction would
  let the agent lose facts nobody chose to lose.
"""

from orion.security.trust import read_text, scan_injection, warn
from orion.tools.base import Tool, ToolError, tool

MEMORY_FILE = ".orion/memory.md"
NOTE_MAX_CHARS = 500
MEMORY_MAX_CHARS = 4_000


def load_memory(env, policy=None) -> str:
    """Return the capped memory text, or a blocked marker if it fails the scan."""
    text = (read_text(env, MEMORY_FILE, policy) or "").strip()
    findings = scan_injection(text)
    if findings:
        warn(f"{MEMORY_FILE} looks like prompt injection: {', '.join(findings)}; not loaded")
        return f"[BLOCKED: {MEMORY_FILE} contained potential prompt injection " \
               f"({', '.join(findings)}). Content not loaded.]"
    return text[:MEMORY_MAX_CHARS]


def remember_tool(env) -> Tool:
    """Build `remember`, which appends one note to the memory file."""

    @tool("Save one durable fact for future sessions: a project convention, a user "
          "preference, a decision. Not for task progress.", kind="write",
          note="The fact, one line, at most 500 chars")
    def remember(note: str) -> str:
        note = " ".join(note.split())  # one line, so a note cannot forge file structure
        if not note:
            raise ToolError("note is empty; pass the fact to remember")
        if len(note) > NOTE_MAX_CHARS:
            raise ToolError(f"note is {len(note)} chars; shorten it to {NOTE_MAX_CHARS}")
        findings = scan_injection(note)
        if findings:
            raise ToolError(f"note rejected ({', '.join(findings)}); memory reloads every "
                            "session, so store plain facts only")
        existing = read_text(env, MEMORY_FILE) or ""
        lines = {line[2:].lower() for line in existing.splitlines() if line.startswith("- ")}
        if note.lower() in lines:
            raise ToolError("that note is already in memory; nothing to do")
        entry = f"- {note}\n"
        if len(existing) + len(entry) > MEMORY_MAX_CHARS:
            raise ToolError(f"memory is full ({MEMORY_MAX_CHARS} chars); the user must prune "
                            f"{MEMORY_FILE}")
        env.write_bytes(MEMORY_FILE, (existing + entry).encode("utf-8"))
        return f"Remembered: {note}"

    return remember

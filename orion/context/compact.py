"""Day 3 — compaction mechanics: where to cut and what the new tape looks like.

Concept: the tape becomes [first user message] + [compaction message] +
[recent tail]. The head pins the task, the summary replaces the middle, and
the tail keeps the live work verbatim.

Design rules:
- The pairing invariant is kept by construction: a `tool` message is never
  a cut point, so the tail cannot begin with an orphan, and a kept assistant
  brings its results along. There is nothing to repair afterwards.
- Pure except for the summary text, which the caller supplies.
- Pruning copies. Nothing here mutates the tape it was given, so a failure
  partway through cannot half-apply.
- Prune rewrites old messages, which ends the provider's cached prefix for
  one turn. That is fine at the threshold and is why it never runs earlier.
"""

import json

from orion.context import summary as summary_mod
from orion.context.tokens import estimate_message

PRUNE_MIN_CHARS = 200
PRESSURE_MIN_CHARS = 2_000
_CLEARED = "[output cleared to save context:"


def valid_cuts(messages: list, start: int) -> list[int]:
    """Indices at or after `start` where a tail may legally begin."""
    return [i for i in range(start, len(messages)) if messages[i]["role"] != "tool"]


def head_len(messages: list) -> int:
    """Messages pinned at the front: the first user message and any compaction."""
    return 2 if len(messages) > 1 and "compaction" in messages[1] else 1


def _last_round(messages: list) -> int:
    """Index of the newest assistant message."""
    return max((i for i, m in enumerate(messages) if m["role"] == "assistant"),
               default=len(messages))


def find_cut(messages: list, keep_tokens: int) -> int:
    """Where the tail starts; `head_len` means there is nothing to summarize."""
    head = head_len(messages)
    total, boundary = 0, None
    for i in range(len(messages) - 1, head - 1, -1):
        total += estimate_message(messages[i])
        if total >= keep_tokens:
            boundary = i
            break
    if boundary is None:
        return head
    newest = _last_round(messages)
    cuts = valid_cuts(messages, boundary)
    # The tail always holds the newest complete round, however large it is.
    return max(min(cuts[0] if cuts else newest, newest), head)


def _calls_by_id(messages: list) -> dict:
    return {c["id"]: c for m in messages for c in m.get("tool_calls") or []}


def _clear(message: dict, call: dict | None) -> dict:
    args = json.dumps(call["args"]) if call else "{}"
    args = args if len(args) <= 120 else args[:117] + "..."
    text = (f"{_CLEARED} {message['name']}({args}), {len(message['text'])} chars. "
            "Re-run if needed.]")
    return {**message, "text": text}


def prune(messages: list, cut: int) -> tuple[list, int]:
    """Clear old tool outputs in `head_len..cut`; return (new tape, reclaimed chars)."""
    calls = _calls_by_id(messages)
    new, reclaimed = list(messages), 0
    for i in range(head_len(messages), cut):
        m = messages[i]
        if m["role"] != "tool" or len(m["text"]) <= PRUNE_MIN_CHARS or m["text"].startswith(_CLEARED):
            continue
        cleared = _clear(m, calls.get(m["tool_call_id"]))
        if len(cleared["text"]) < len(m["text"]):  # the marker can outgrow a short result
            new[i] = cleared
            reclaimed += len(m["text"]) - len(cleared["text"])
    return new, reclaimed


def prune_pressure(messages: list) -> tuple[list, int]:
    """Overflow only: clip large results in the tail too, sparing the newest round."""
    new, reclaimed = list(messages), 0
    for i in range(head_len(messages), _last_round(messages)):
        m = messages[i]
        if m["role"] == "tool" and len(m["text"]) > PRESSURE_MIN_CHARS:
            kept = f"{m['text'][:PRESSURE_MIN_CHARS]}\n... [clipped under context pressure]"
            new[i] = {**m, "text": kept}
            reclaimed += len(m["text"]) - len(kept)
    return new, reclaimed


def previous_details(messages: list) -> dict | None:
    """The structured details of the last compaction, read from the field, never the text."""
    return messages[1]["compaction"] if head_len(messages) == 2 else None


def build_details(messages: list, cut: int, summary: str, tools: dict, degraded: bool = False) -> dict:
    """Compute the compaction details for folding `head_len..cut`."""
    previous = previous_details(messages) or {}
    span = messages[head_len(messages):cut]
    read, modified = summary_mod.extract_file_ops(span, tools, previous)
    return {
        "summary": summary,
        "user_messages": summary_mod.extract_user_messages(span, previous.get("user_messages")),
        "read_files": read,
        "modified_files": modified,
        "skills": summary_mod.extract_skills(span, previous.get("skills")),
        "count": previous.get("count", 0) + 1,
        "degraded": degraded,
    }


def apply(messages: list, cut: int, details: dict) -> list:
    """Compose the new tape: head, one compaction message, verbatim tail."""
    compaction = {"role": "user", "text": summary_mod.render(details), "compaction": details}
    return [messages[0], compaction] + messages[cut:]

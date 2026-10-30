"""Day 3 — token accounting: measure with real numbers, estimate only the rest.

Concept: the provider reports the true size of every prompt it receives, so
the last report is an anchor. Only what was appended since is estimated.

Design rules:
- Estimates are deliberately rough (4 chars per token); they cover a few
  messages between anchors, not the whole tape.
- `thoughts` are billed but not re-sent, so they stay out of the anchor.
- Every prompt token is paid again on every turn, which is why a 1M window
  still compacts at 200k. That is also where Pro pricing doubles.
"""

import json
import math

CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD = 10
OVERHEAD_MARGIN = 1.2

RESERVE_TOKENS = 16_384        # space held back for the reply
KEEP_RECENT_TOKENS = 20_000    # verbatim tail
MAX_THRESHOLD = 200_000        # a huge window must not mean huge per-turn cost


def estimate_message(m: dict) -> int:
    """Rough tokens for one neutral message."""
    # DeepSeek re-reads its reasoning, which is stored in text_signature.
    chars = len(m.get("text") or "") + len(m.get("text_signature") or "")
    if m.get("tool_calls"):
        chars += len(json.dumps(m["tool_calls"]))
    return math.ceil(chars / CHARS_PER_TOKEN) + MESSAGE_OVERHEAD


def estimate(messages: list) -> int:
    """Rough tokens for a list of neutral messages."""
    return sum(estimate_message(m) for m in messages)


def estimate_overhead(system: str, tool_specs: list) -> int:
    """Rough tokens for the system prompt and tool schemas, with a safety margin."""
    chars = len(system) + len(json.dumps([t["schema"] for t in tool_specs]))
    return math.ceil(chars / CHARS_PER_TOKEN * OVERHEAD_MARGIN)


def compaction_threshold(window: int) -> int:
    """The prompt size at which compaction starts for a given window."""
    return min(int(0.75 * (window - RESERVE_TOKENS)), MAX_THRESHOLD)


def keep_recent(threshold: int) -> int:
    """Tokens of verbatim tail to keep for a given threshold."""
    return min(KEEP_RECENT_TOKENS, threshold // 4)


class TokenMeter:
    """The usage anchor: real size of the last prompt, plus an estimate of the rest."""

    def __init__(self, overhead: int = 0):
        """`overhead` is the estimated system-and-tools size, used until the first anchor."""
        self.overhead = overhead
        self.anchor = None
        self.anchor_len = 0

    def record(self, usage: dict, tape_len: int) -> None:
        """Anchor on a reply's usage; `tape_len` must count that reply's message."""
        # `input` already includes cached tokens: they are a discount, not extra.
        self.anchor = usage["input"] + usage["output"]
        self.anchor_len = tape_len

    def current(self, messages: list) -> int:
        """Best estimate of the next prompt's size."""
        if self.anchor is None:
            return self.overhead + estimate(messages)
        return self.anchor + estimate(messages[self.anchor_len:])

    def reset(self, overhead: int) -> None:
        """Forget the anchor after the tape changed; estimate until the next real usage."""
        self.overhead = overhead
        self.anchor = None
        self.anchor_len = 0

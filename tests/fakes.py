"""Day 1 — test fakes: a scripted stand-in for the model.

Concept: the loop takes its model call as a parameter, so tests replace the
network with a script and stay offline and deterministic.

Design rules:
- Same signature as `provider.complete`.
- Every `messages` list handed in is deep-copied, so tests see exactly what
  the model saw at each call, not the later mutated tape.
- Running out of script is a test failure, never a silent default.
"""

import copy


class ScriptedProvider:
    """Callable fake model that returns scripted replies in order."""

    def __init__(self, replies):
        """Store the replies: provider-shaped dicts, returned in order."""
        self.replies = list(replies)
        self.calls = []  # (messages snapshot, tools) per call

    @property
    def seen(self):
        """The messages lists the fake was handed, one per call."""
        return [messages for messages, _ in self.calls]

    def __call__(self, model, system, messages, tools):
        """Record the call and return the next scripted reply."""
        self.calls.append((copy.deepcopy(messages), list(tools)))
        if len(self.calls) > len(self.replies):
            raise AssertionError(f"ScriptedProvider ran out of replies at call {len(self.calls)}")
        return self.replies[len(self.calls) - 1]


def reply(text="", tool_calls=(), usage=None):
    """Build a provider-shaped reply dict for scripts."""
    return {
        "text": text,
        "tool_calls": list(tool_calls),
        "text_signature": None,
        "finish": "STOP",
        "usage": usage or {"input": 10, "output": 5, "thoughts": 0, "cached": 0},
    }


def call(name, call_id, **args):
    """Build a neutral ToolCall dict."""
    return {"id": call_id, "name": name, "args": args, "signature": None}

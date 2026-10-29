"""Day 1 — live demo: one tool, one loop, a real model.

Concept: the smallest end-to-end agent run. The model decides to call
`roll_dice`, the loop runs it, pairs the result to the call by id, and the
model answers using the result.

Design rules:
- The demo owns the tool and the printing; the loop knows neither.
- Tool errors are raised, not returned, so the loop's error containment is
  what reports them to the model.
- `--trace` dumps the full neutral tape so the call/result pairing is visible.
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orion import provider  # noqa: E402
from orion.loop import run_loop  # noqa: E402

DEFAULT_TASK = "Roll 3 dice and tell me whether the total beats 10"
SYSTEM = "You are Orion, a helpful agent. Use tools when they help; answer concisely."


class RollDice:
    """A tool that rolls N six-sided dice."""

    spec = {"schema": {
        "name": "roll_dice",
        "description": "Roll N six-sided dice. Returns the individual rolls and their total.",
        "parameters": {
            "type": "object",
            "properties": {"count": {"type": "integer", "description": "How many dice, 1 to 10"}},
            "required": ["count"],
        },
    }}

    def run(self, count):
        """Roll `count` dice (1-10) and return the rolls and total as text."""
        count = int(count)
        if not 1 <= count <= 10:
            raise ValueError(f"count must be between 1 and 10, got {count}")
        rolls = [random.randint(1, 6) for _ in range(count)]
        return f"rolls: {rolls} total: {sum(rolls)}"


def print_event(kind, payload):
    """Print loop events as a readable transcript."""
    if kind == "assistant" and payload["text"]:
        print(f"assistant: {payload['text']}")
    elif kind == "tool_start":
        print(f"  -> {payload['name']}({json.dumps(payload['args'])})  [{payload['id']}]")
    elif kind == "tool_end":
        print(f"  <- {payload['result']}")
    elif kind == "turn_end":
        u, t = payload["usage"], payload["total"]
        print(f"[turn {payload['turn']}] in={u['input']} out={u['output']} "
              f"| total in={t['input']} out={t['output']}")


def main(argv):
    """Run the demo task from argv (or the default) and print the outcome."""
    trace = "--trace" in argv
    args = [a for a in argv if a != "--trace"]
    task = " ".join(args) or DEFAULT_TASK

    messages = [{"role": "user", "text": task}]
    print(f"user: {task}")
    result = run_loop(provider.default_model(), SYSTEM, messages,
                      {"roll_dice": RollDice()}, on_event=print_event)

    if trace:
        print(json.dumps(messages, indent=2))
    print(f"\nfinal: {result.text}")
    print(f"stop={result.stop} turns={result.turns} usage={result.usage}")


if __name__ == "__main__":
    main(sys.argv[1:])

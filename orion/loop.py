"""Day 1 — the loop: the heartbeat of the harness.

Concept: an agent is a loop. Call the model; if it asks for tools, run them,
append the results, and call it again; stop when it answers without tools or
a guard trips.

Design rules:
- The loop never knows tool names; tools arrive as a dict of objects with
  `.spec` and `.run`.
- `messages` is mutated in place so the caller always holds the live tape.
- Pairing invariant: every tool call is answered exactly once — allowed,
  blocked, unknown or failed. A tool exception never crashes the loop.
- Never end a run without a report: when a guard trips, ask for a handover
  and make one final call with no tools.
- The model call is injectable (`complete=`) so tests run offline.
"""

from dataclasses import dataclass

from orion import provider

HANDOVER_PROMPT = (
    "Turn or token limit reached. Stop working and write a handover: the "
    "original task; what is finished and verified; what is half-done and "
    "exactly where; the next concrete step."
)

_USAGE_KEYS = ("input", "output", "thoughts", "cached")


@dataclass
class RunResult:
    """Outcome of one run_loop call."""

    text: str
    stop: str  # "done" | "max_turns" | "budget"
    turns: int
    usage: dict  # summed input/output/thoughts/cached across all calls


def run_loop(model, system, messages, tools, on_event=None, before_tool=None,
             before_turn=None, max_turns=80, max_tokens=None,
             complete=None) -> RunResult:
    """Run the agent loop until the model answers without tools or a guard trips.

    `tools` maps name -> Tool (with `.spec` and `.run`). `on_event(kind,
    payload)` observes progress; `before_tool(call)` returns a reason string
    to block a call; `before_turn(messages)` may rewrite the tape before each
    model call. Returns a RunResult; `messages` holds the full tape.
    """
    complete = complete or provider.complete
    emit = on_event or (lambda kind, payload: None)
    specs = [t.spec for t in tools.values()]
    total = {k: 0 for k in _USAGE_KEYS}
    turns = 0

    def call_model(tool_specs):
        nonlocal turns
        if before_turn is not None:
            messages[:] = before_turn(messages)
        reply = complete(model, system, messages, tool_specs)
        turns += 1
        msg = {
            "role": "assistant",
            "text": reply["text"],
            "tool_calls": reply["tool_calls"],
            "text_signature": reply.get("text_signature"),
        }
        messages.append(msg)
        emit("assistant", msg)
        for k in _USAGE_KEYS:
            total[k] += reply["usage"].get(k, 0)
        emit("turn_end", {"turn": turns, "usage": reply["usage"], "total": dict(total)})
        return msg

    while True:
        msg = call_model(specs)
        if not msg["tool_calls"]:
            return RunResult(msg["text"], "done", turns, total)

        for call in msg["tool_calls"]:
            emit("tool_start", call)
            result = _run_tool(call, tools, before_tool)
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["name"],
                "text": result,
            })
            emit("tool_end", {"call": call, "result": result})

        if turns >= max_turns:
            stop = "max_turns"
        elif max_tokens is not None and total["input"] + total["output"] > max_tokens:
            stop = "budget"
        else:
            continue
        # Guards are checked only after every call is answered, so the
        # handover request never interrupts a half-paired turn.
        messages.append({"role": "user", "text": HANDOVER_PROMPT})
        final = call_model([])
        return RunResult(final["text"], stop, turns, total)


def _run_tool(call, tools, before_tool) -> str:
    """Run one tool call and return its result text; never raises."""
    if before_tool is not None:
        reason = before_tool(call)
        if reason is not None:
            return f"BLOCKED: {reason}"
    tool = tools.get(call["name"])
    if tool is None:
        return f"ERROR: unknown tool {call['name']}"
    try:
        return str(tool.run(**call["args"]))
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def check_pairing(messages) -> list:
    """Return a list of pairing problems in the tape (empty when legal).

    Problems: a tool message answering no earlier unanswered call, a call
    answered twice, or a call left unanswered when a later assistant message
    follows.
    """
    problems = []
    pending = {}  # call id -> tool name, for calls not yet answered
    answered = set()
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            for call_id, name in pending.items():
                problems.append(f"message {i}: call {call_id} ({name}) was never answered")
            pending = {c["id"]: c["name"] for c in m.get("tool_calls") or []}
        elif m["role"] == "tool":
            call_id = m.get("tool_call_id")
            if call_id in pending:
                del pending[call_id]
                answered.add(call_id)
            elif call_id in answered:
                problems.append(f"message {i}: call {call_id} answered twice")
            else:
                problems.append(f"message {i}: tool result {call_id} matches no earlier call")
    return problems

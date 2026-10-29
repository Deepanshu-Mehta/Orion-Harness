"""Day 1 — tests for the agent loop, fully offline.

Concept: the loop's contract (stop reasons, tool result pairing, error
containment, guards, usage totals) is proved against a scripted model.

Design rules: no network, no API key; every test ends by checking the tape
still satisfies the pairing invariant.
"""

import unittest

from fakes import ScriptedProvider, call, reply
from orion.loop import HANDOVER_PROMPT, check_pairing, run_loop


class Tool:
    """Minimal Tool: a spec plus a run callable."""

    def __init__(self, name, fn):
        self.spec = {"schema": {"name": name, "description": name,
                                "parameters": {"type": "object", "properties": {}}}}
        self.run = fn


def echo(**kwargs):
    """Return the arguments as text."""
    return f"echo {kwargs}"


def explode(**kwargs):
    """Always fail."""
    raise ValueError("bad input")


TOOLS = {"echo": Tool("echo", echo), "explode": Tool("explode", explode)}


class LoopTests(unittest.TestCase):
    def run_script(self, replies, **kwargs):
        fake = ScriptedProvider(replies)
        messages = [{"role": "user", "text": "go"}]
        result = run_loop("m", "sys", messages, kwargs.pop("tools", TOOLS),
                          complete=fake, **kwargs)
        self.assertEqual(check_pairing(messages), [])
        return result, messages, fake

    def tool_messages(self, messages):
        return [m for m in messages if m["role"] == "tool"]

    def test_plain_answer(self):
        result, messages, _ = self.run_script([reply("hello")])
        self.assertEqual((result.text, result.stop, result.turns), ("hello", "done", 1))
        self.assertEqual(len(messages), 2)

    def test_one_tool_call(self):
        result, messages, fake = self.run_script([
            reply(tool_calls=[call("echo", "c1", x=1)]),
            reply("done"),
        ])
        tool_msgs = self.tool_messages(messages)
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["tool_call_id"], "c1")
        self.assertEqual(tool_msgs[0]["text"], "echo {'x': 1}")
        self.assertEqual((result.text, result.stop, result.turns), ("done", "done", 2))
        # The second model call saw the tool result.
        self.assertEqual(fake.seen[1][-1]["tool_call_id"], "c1")

    def test_parallel_calls_keep_order(self):
        calls = [call("echo", f"c{i}", i=i) for i in range(3)]
        _, messages, _ = self.run_script([reply(tool_calls=calls), reply("ok")])
        tool_msgs = self.tool_messages(messages)
        self.assertEqual([m["tool_call_id"] for m in tool_msgs], ["c0", "c1", "c2"])
        self.assertEqual([m["text"] for m in tool_msgs],
                         [f"echo {{'i': {i}}}" for i in range(3)])

    def test_unknown_tool(self):
        result, messages, _ = self.run_script([
            reply(tool_calls=[call("nope", "c1")]),
            reply("recovered"),
        ])
        self.assertEqual(self.tool_messages(messages)[0]["text"], "ERROR: unknown tool nope")
        self.assertEqual(result.stop, "done")

    def test_tool_raises(self):
        result, messages, _ = self.run_script([
            reply(tool_calls=[call("explode", "c1")]),
            reply("recovered"),
        ])
        self.assertEqual(self.tool_messages(messages)[0]["text"], "ERROR: ValueError: bad input")
        self.assertEqual(result.text, "recovered")

    def test_before_tool_blocks(self):
        ran = []
        tools = {"echo": Tool("echo", lambda **kw: ran.append(kw) or "ran")}
        _, messages, _ = self.run_script(
            [reply(tool_calls=[call("echo", "c1")]), reply("ok")],
            tools=tools, before_tool=lambda c: "not allowed",
        )
        tool_msgs = self.tool_messages(messages)
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["text"], "BLOCKED: not allowed")
        self.assertEqual(ran, [])

    def test_max_turns_wrap_up(self):
        result, messages, fake = self.run_script(
            [reply(tool_calls=[call("echo", "c1")]), reply("handover notes")],
            max_turns=1,
        )
        self.assertEqual(result.stop, "max_turns")
        self.assertEqual(result.text, "handover notes")
        tools_per_call = [tools for _, tools in fake.calls]
        self.assertEqual(len(tools_per_call[0]), 2)
        self.assertEqual(tools_per_call[1], [])
        self.assertEqual(fake.seen[1][-1], {"role": "user", "text": HANDOVER_PROMPT})
        self.assertEqual(messages[-1]["text"], "handover notes")

    def test_token_budget_wrap_up(self):
        result, _, fake = self.run_script(
            [reply(tool_calls=[call("echo", "c1")]), reply("handover")],
            max_tokens=10,
        )
        self.assertEqual(result.stop, "budget")
        self.assertEqual(fake.calls[1][1], [])

    def test_usage_sums_across_turns(self):
        u1 = {"input": 100, "output": 20, "thoughts": 7, "cached": 3}
        u2 = {"input": 150, "output": 30, "thoughts": 1, "cached": 50}
        events = []
        result, _, _ = self.run_script(
            [reply(tool_calls=[call("echo", "c1")], usage=u1), reply("ok", usage=u2)],
            on_event=lambda kind, payload: events.append((kind, payload)),
        )
        self.assertEqual(result.usage, {"input": 250, "output": 50, "thoughts": 8, "cached": 53})
        turn_ends = [p for k, p in events if k == "turn_end"]
        self.assertEqual([p["turn"] for p in turn_ends], [1, 2])
        self.assertEqual(turn_ends[0]["usage"], u1)
        self.assertEqual(turn_ends[1]["total"], result.usage)
        kinds = [k for k, _ in events]
        self.assertEqual(kinds, ["assistant", "turn_end", "tool_start", "tool_end",
                                 "assistant", "turn_end"])

    def test_before_turn_replaces_in_place(self):
        fake = ScriptedProvider([reply("ok")])
        messages = [{"role": "user", "text": "old"}]
        tape = messages
        run_loop("m", "sys", messages, {}, complete=fake,
                 before_turn=lambda ms: [{"role": "user", "text": "new"}])
        self.assertIs(messages, tape)
        self.assertEqual(fake.seen[0], [{"role": "user", "text": "new"}])


class PairingTests(unittest.TestCase):
    def assistant(self, *ids):
        return {"role": "assistant", "text": "",
                "tool_calls": [call("echo", i) for i in ids], "text_signature": None}

    def tool(self, call_id):
        return {"role": "tool", "tool_call_id": call_id, "name": "echo", "text": "r"}

    def test_dangling_call(self):
        tape = [{"role": "user", "text": "go"}, self.assistant("c1", "c2"),
                self.tool("c1"), self.assistant()]
        problems = check_pairing(tape)
        self.assertEqual(len(problems), 1)
        self.assertIn("c2", problems[0])

    def test_double_answer(self):
        tape = [self.assistant("c1"), self.tool("c1"), self.tool("c1")]
        self.assertIn("answered twice", check_pairing(tape)[0])

    def test_orphan_result(self):
        tape = [{"role": "user", "text": "go"}, self.tool("ghost")]
        self.assertIn("matches no earlier call", check_pairing(tape)[0])

    def test_trailing_unanswered_call_is_legal(self):
        # A call still awaiting its result (no later assistant message) is
        # an in-flight turn, not a violation.
        self.assertEqual(check_pairing([self.assistant("c1")]), [])


if __name__ == "__main__":
    unittest.main()

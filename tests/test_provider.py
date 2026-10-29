"""Day 1 — tests for the provider's pure translation layer.

Concept: the wire format rules that Gemini enforces (result grouping,
signature echo, thought filtering) are checked without touching the network.

Design rules: call only the pure functions `_to_wire` and `_parse`, plus
`api_key()` under a patched environment.
"""

import os
import unittest
from unittest import mock

from orion import provider


def tool_msg(call_id, text):
    return {"role": "tool", "tool_call_id": call_id, "name": "roll", "text": text}


class ToWireTests(unittest.TestCase):
    def test_groups_consecutive_tool_messages(self):
        messages = [
            {"role": "user", "text": "go"},
            {"role": "assistant", "text": "", "text_signature": None, "tool_calls": [
                {"id": "a", "name": "roll", "args": {"n": 1}, "signature": None},
                {"id": "b", "name": "roll", "args": {"n": 2}, "signature": None},
            ]},
            tool_msg("a", "one"),
            tool_msg("b", "two"),
            {"role": "user", "text": "wrap up"},
        ]
        wire = provider._to_wire(messages)
        self.assertEqual([c["role"] for c in wire], ["user", "model", "user"])
        results = wire[2]["parts"]
        self.assertEqual(
            [p["functionResponse"]["response"]["result"] for p in results[:2]], ["one", "two"])
        self.assertEqual(results[2], {"text": "wrap up"})

    def test_signatures_echo(self):
        messages = [{"role": "assistant", "text": "thinking aloud", "text_signature": "tsig",
                     "tool_calls": [
                         {"id": "a", "name": "roll", "args": {}, "signature": "sig-a"},
                         {"id": "b", "name": "roll", "args": {}, "signature": None},
                     ]}]
        parts = provider._to_wire(messages)[0]["parts"]
        self.assertEqual(parts[0], {"text": "thinking aloud", "thoughtSignature": "tsig"})
        self.assertEqual(parts[1]["thoughtSignature"], "sig-a")
        self.assertEqual(parts[1]["functionCall"], {"name": "roll", "args": {}})
        self.assertNotIn("thoughtSignature", parts[2])

    def test_empty_text_omits_text_part(self):
        messages = [{"role": "assistant", "text": "", "text_signature": None,
                     "tool_calls": [{"id": "a", "name": "roll", "args": {}, "signature": None}]}]
        parts = provider._to_wire(messages)[0]["parts"]
        self.assertEqual(len(parts), 1)
        self.assertIn("functionCall", parts[0])


class ParseTests(unittest.TestCase):
    def response(self, parts, usage=None, finish="STOP"):
        resp = {"candidates": [{"content": {"role": "model", "parts": parts},
                                "finishReason": finish}]}
        if usage is not None:
            resp["usageMetadata"] = usage
        return resp

    def test_skips_thoughts_keeps_text_signature(self):
        out = provider._parse(self.response([
            {"text": "secret reasoning", "thought": True, "thoughtSignature": "thought-sig"},
            {"text": "Hello ", "thoughtSignature": "s1"},
            {"text": "world", "thoughtSignature": "s2"},
        ]))
        self.assertEqual(out["text"], "Hello world")
        self.assertEqual(out["text_signature"], "s2")
        self.assertEqual(out["finish"], "STOP")

    def test_function_calls_get_ids_and_signatures(self):
        out = provider._parse(self.response([
            {"functionCall": {"name": "roll", "args": {"count": 3}}, "thoughtSignature": "sig"},
            {"functionCall": {"name": "roll", "args": {"count": 1}}},
            {"functionCall": {"name": "roll", "args": {}, "id": "vendor-id"}},
        ]))
        calls = out["tool_calls"]
        self.assertEqual(calls[0]["signature"], "sig")
        self.assertIsNone(calls[1]["signature"])
        self.assertEqual(calls[2]["id"], "vendor-id")
        generated = [c["id"] for c in calls[:2]]
        for cid in generated:
            self.assertRegex(cid, r"^call_[0-9a-f]{12}$")
        self.assertEqual(len(set(generated)), 2)

    def test_missing_usage_defaults_to_zero(self):
        out = provider._parse(self.response([{"text": "hi"}], usage={"promptTokenCount": 7}))
        self.assertEqual(out["usage"], {"input": 7, "output": 0, "thoughts": 0, "cached": 0})
        out = provider._parse(self.response([{"text": "hi"}]))
        self.assertEqual(out["usage"], {"input": 0, "output": 0, "thoughts": 0, "cached": 0})

    def test_no_candidates_raises_with_feedback(self):
        with self.assertRaises(RuntimeError) as ctx:
            provider._parse({"promptFeedback": {"blockReason": "SAFETY"}})
        self.assertIn("SAFETY", str(ctx.exception))


class EnvTests(unittest.TestCase):
    def test_api_key_missing(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                provider.api_key()
        self.assertIn("ORION_API_KEY", str(ctx.exception))
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    def test_api_key_fallback_order(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "g"}, clear=True):
            self.assertEqual(provider.api_key(), "g")
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "g", "ORION_API_KEY": "o"}, clear=True):
            self.assertEqual(provider.api_key(), "o")

    def test_default_model(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(provider.default_model(), provider.DEFAULT_MODEL)
        with mock.patch.dict(os.environ, {"ORION_MODEL": "x"}, clear=True):
            self.assertEqual(provider.default_model(), "x")


if __name__ == "__main__":
    unittest.main()

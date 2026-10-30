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


class DeepSeekTests(unittest.TestCase):
    TAPE = [
        {"role": "user", "text": "first"},
        {"role": "user", "text": "compaction summary"},
        {"role": "assistant", "text": "", "text_signature": "reasoning one", "tool_calls": [
            {"id": "a", "name": "roll", "args": {"n": 1}, "signature": None}]},
        tool_msg("a", "one"),
        {"role": "assistant", "text": "done", "text_signature": "reasoning two", "tool_calls": []},
    ]

    def test_wire_shape(self):
        wire = provider._to_deepseek("sys", self.TAPE)
        self.assertEqual(wire[0], {"role": "system", "content": "sys"})
        self.assertEqual(wire[1], {"role": "user", "content": "first\n\ncompaction summary"})
        self.assertEqual(wire[2]["tool_calls"], [{"id": "a", "type": "function", "function": {
            "name": "roll", "arguments": '{"n": 1}'}}])
        self.assertEqual(wire[3], {"role": "tool", "tool_call_id": "a", "content": "one"})
        self.assertNotIn("tool_calls", wire[4])

    def test_reasoning_is_echoed_on_every_turn(self):
        wire = provider._to_deepseek("sys", self.TAPE)
        self.assertEqual([m.get("reasoning_content") for m in wire if m["role"] == "assistant"],
                         ["reasoning one", "reasoning two"])

    def response(self, message, finish="stop", usage=None):
        return {"choices": [{"message": message, "finish_reason": finish}], "usage": usage or {}}

    def test_parse(self):
        out = provider._parse_deepseek(self.response(
            {"content": "hi", "reasoning_content": "because", "tool_calls": [
                {"id": "x", "function": {"name": "roll", "arguments": '{"n": 2}'}}]},
            "tool_calls", {"prompt_tokens": 100, "completion_tokens": 30,
                           "prompt_cache_hit_tokens": 60,
                           "completion_tokens_details": {"reasoning_tokens": 20}}))
        self.assertEqual(out["tool_calls"], [{"id": "x", "name": "roll", "args": {"n": 2},
                                              "signature": None}])
        self.assertEqual(out["text_signature"], "because")
        self.assertEqual(out["finish"], "TOOL_CALLS")
        self.assertEqual(out["usage"], {"input": 100, "output": 10, "thoughts": 20, "cached": 60})

    def test_finish_maps_to_the_neutral_names(self):
        self.assertEqual(provider._parse_deepseek(self.response({}, "length"))["finish"], "MAX_TOKENS")
        self.assertEqual(provider._parse_deepseek(self.response({}, "stop"))["finish"], "STOP")

    def test_bad_arguments_become_a_rejectable_argument(self):
        out = provider._parse_deepseek(self.response({"tool_calls": [
            {"id": "x", "function": {"name": "roll", "arguments": "{broken"}}]}))
        self.assertEqual(out["tool_calls"][0]["args"], {"invalid_json_arguments": "{broken"})

    def test_no_choices_raises(self):
        with self.assertRaises(RuntimeError):
            provider._parse_deepseek({"choices": []})

    def test_limits_need_no_network(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch("urllib.request.urlopen", side_effect=AssertionError("network")):
            self.assertEqual(provider.context_window("deepseek-flash"), 1_000_000)

    def test_keys_only_reach_their_own_vendor(self):
        keys = {"GEMINI_API_KEY": "gem-secret", "DEEPSEEK_API_KEY": "ds-secret"}
        with mock.patch.dict(os.environ, keys, clear=True), \
                mock.patch("orion.provider._request", return_value={"choices": [{"message": {}}]}) as sent:
            provider.complete("deepseek-flash", "s", [{"role": "user", "text": "x"}], [])
        self.assertEqual(sent.call_args.kwargs["headers"]["Authorization"], "Bearer ds-secret")
        self.assertNotIn("gem-secret", str(sent.call_args))
        with mock.patch.dict(os.environ, keys, clear=True), \
                mock.patch("urllib.request.urlopen", side_effect=TimeoutError("t")) as opened, \
                mock.patch("time.sleep"):
            with self.assertRaises(RuntimeError):
                provider.complete("gemini-x", "s", [{"role": "user", "text": "x"}], [])
        self.assertNotIn("ds-secret", str(opened.call_args[0][0].headers))

    def test_missing_key_names_the_variable(self):
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError) as ctx:
            provider.complete("deepseek-flash", "s", [], [])
        self.assertIn("DEEPSEEK_API_KEY", str(ctx.exception))

    def test_web_search_stays_on_gemini(self):
        with mock.patch.dict(os.environ, {"ORION_MODEL": "deepseek-flash"}, clear=True), \
                mock.patch("orion.provider._post", return_value={
                    "candidates": [{"content": {"parts": [{"text": "a"}]}}]}) as post:
            provider.web_search("q")
        self.assertIn(provider.GROUNDING_MODEL, post.call_args[0][0])


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

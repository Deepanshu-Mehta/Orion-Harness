"""Day 3 — tests for token accounting and model limits.

Concept: the meter trusts the provider's numbers and estimates only what
came after them; overflow errors and window lookups fail in safe directions.

Design rules: offline; HTTP is faked at the urllib boundary.
"""

import email.message
import io
import os
import unittest
import urllib.error
from unittest import mock

from orion import provider
from orion.context import tokens
from orion.context.tokens import TokenMeter


def user(text):
    return {"role": "user", "text": text}


class MeterTests(unittest.TestCase):
    def test_estimates(self):
        self.assertEqual(tokens.estimate_message(user("x" * 40)), 20)
        self.assertEqual(tokens.estimate_message(user("x" * 41)), 21)
        call = {"role": "assistant", "text": "", "tool_calls": [{"id": "a"}]}
        self.assertGreater(tokens.estimate_message(call), 10)
        self.assertEqual(tokens.estimate_overhead("x" * 400, []), 120 + 1)

    def test_anchor_plus_trailing_estimate(self):
        meter = TokenMeter()
        tape = [user("a"), user("b")]
        meter.record({"input": 1000, "output": 50, "thoughts": 900, "cached": 800}, len(tape))
        self.assertEqual(meter.current(tape), 1050)  # thoughts excluded, cached not added
        tape.append(user("x" * 400))
        self.assertEqual(meter.current(tape), 1050 + 110)

    def test_no_anchor_uses_overhead(self):
        meter = TokenMeter(overhead=500)
        self.assertEqual(meter.current([user("x" * 40)]), 520)

    def test_reset_falls_back_to_estimate(self):
        meter = TokenMeter()
        meter.record({"input": 9000, "output": 1}, 3)
        meter.reset(100)
        self.assertEqual(meter.current([user("x" * 40)]), 120)

    def test_thresholds(self):
        self.assertEqual(tokens.compaction_threshold(1_000_000), 200_000)
        self.assertEqual(tokens.compaction_threshold(200_000), 137_712)
        self.assertEqual(tokens.compaction_threshold(128_000), 83_712)
        self.assertEqual(tokens.compaction_threshold(32_000), 11_712)
        self.assertEqual(tokens.keep_recent(200_000), 20_000)
        self.assertEqual(tokens.keep_recent(11_712), 2_928)


def http_error(code, body):
    return urllib.error.HTTPError("u", code, "x", email.message.Message(), io.BytesIO(body.encode()))


class ProviderLimitTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        provider._limits_cache.clear()
        self.addCleanup(provider._limits_cache.clear)
        provider._warned_fallback = False

    def test_overflow_from_fixture_body(self):
        body = '{"error":{"code":400,"message":"The input token count (2000000) exceeds the ' \
               'maximum number of tokens allowed (1048576)."}}'
        with mock.patch("urllib.request.urlopen", side_effect=http_error(400, body)):
            with self.assertRaises(provider.ContextOverflow):
                provider._post("https://x.test/", {})

    def test_other_400_stays_runtime_error(self):
        with mock.patch("urllib.request.urlopen", side_effect=http_error(400, "bad field 'x'")):
            with self.assertRaises(RuntimeError) as ctx:
                provider._post("https://x.test/", {})
        self.assertNotIsInstance(ctx.exception, provider.ContextOverflow)

    def test_final_failure_names_the_reason(self):
        body = '{"error":{"message":"exceeded its monthly spending cap"}}'
        errors = [http_error(429, body) for _ in range(5)]
        with mock.patch("urllib.request.urlopen", side_effect=errors), mock.patch("time.sleep"):
            with self.assertRaises(RuntimeError) as ctx:
                provider._post("https://x.test/", {})
        self.assertIn("spending cap", str(ctx.exception))

    def test_limits_cached_and_shared_get(self):
        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        payload = b'{"inputTokenLimit": 1000, "outputTokenLimit": 50}'
        with mock.patch("urllib.request.urlopen", return_value=Resp(payload)) as opened:
            self.assertEqual(provider.model_limits("m"), {"input": 1000, "output": 50})
            self.assertEqual(provider.context_window("m"), 1000)
        self.assertEqual(opened.call_count, 1)
        req = opened.call_args[0][0]
        self.assertEqual(req.get_method(), "GET")
        self.assertEqual(req.get_header("X-goog-api-key"), "k")
        self.assertEqual(req.full_url, f"{provider.API_ROOT}/m")  # no key in the URL

    def test_window_env_override(self):
        with mock.patch.dict(os.environ, {"ORION_CONTEXT_WINDOW": "32000"}):
            self.assertEqual(provider.context_window("m"), 32000)

    def test_window_fallback_warns_once(self):
        err = io.StringIO()
        with mock.patch("orion.provider.model_limits", side_effect=RuntimeError("boom")), \
                mock.patch("sys.stderr", err):
            self.assertEqual(provider.context_window("m"), provider.FALLBACK_WINDOW)
            self.assertEqual(provider.context_window("m"), provider.FALLBACK_WINDOW)
        self.assertEqual(err.getvalue().count("warning"), 1)


if __name__ == "__main__":
    unittest.main()

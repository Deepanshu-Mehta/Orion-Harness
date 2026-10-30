"""Day 3 — tests for the context engine, driven through the real loop.

Concept: the escalation (nothing, prune, summarize), the failure paths
(never lose history), the anti-thrash breaker, overflow recovery, and the
accounting the loop reports.

Design rules: offline. The model is a scripted fake with real usage numbers.
"""

import io
import unittest
from unittest import mock

from fakes import ScriptedProvider, call, reply
from orion import provider
from orion.context import summary
from orion.context.engine import ContextEngine
from orion.context.tokens import estimate_overhead
from orion.loop import check_pairing, run_loop

GOOD = "\n".join(f"{h}\n(none)" for h in summary.HEADINGS)


def usage(prompt, out=10):
    return {"input": prompt, "output": out, "thoughts": 3, "cached": 0}


def summary_reply(text=GOOD, finish="STOP"):
    return {"text": text, "tool_calls": [], "text_signature": None, "finish": finish,
            "usage": {"input": 100, "output": 40, "thoughts": 5, "cached": 2}}


class Echo:
    spec = {"schema": {"name": "echo", "description": "d", "parameters": {"type": "object"}}}
    kind, path_args = "read", {}

    @staticmethod
    def run(**kw):
        return "r" * 4000


def tape(rounds, text_size=4000):
    out = [{"role": "user", "text": "the task"}]
    for i in range(rounds):
        out.append({"role": "assistant", "text": "", "text_signature": None,
                    "tool_calls": [call("echo", f"c{i}")]})
        out.append({"role": "tool", "tool_call_id": f"c{i}", "name": "echo", "text": "r" * text_size})
    return out


class Base(unittest.TestCase):
    def setUp(self):
        err = mock.patch("sys.stderr", io.StringIO())
        self.stderr = err.start()
        self.addCleanup(err.stop)
        self.records, self.builds = [], 0

    def build(self):
        self.builds += 1
        return f"system v{self.builds}"

    def engine(self, summarize=None, complete=None, threshold=1000, keep=100, **kw):
        return ContextEngine("main", self.build, [], window=100_000, threshold=threshold,
                             keep_tokens=keep, summarize=summarize, complete=complete,
                             on_compaction=self.records.append, **kw)


class EscalationTests(Base):
    def test_under_threshold_untouched(self):
        e = self.engine(summarize=mock.Mock())
        t = tape(1)
        self.assertIs(e.before_turn(t), t)
        e._summarize.assert_not_called()

    def test_prune_alone_costs_no_model_call(self):
        e = self.engine(summarize=mock.Mock(), threshold=4000, keep=500)
        t = tape(6)
        new = e.before_turn(t)
        e._summarize.assert_not_called()
        self.assertEqual(len(new), len(t))
        self.assertEqual(self.records[0]["method"], "prune")
        self.assertIn("output cleared", new[2]["text"])
        self.assertEqual(check_pairing(new), [])
        self.assertIsNone(e.meter.anchor)  # the anchor described the unpruned tape
        self.assertEqual(self.builds, 1)   # prompt untouched by a prune

    def test_summary_sees_unpruned_output(self):
        seen = []

        def summarize(system, messages):
            seen.append(messages[0]["text"])
            return summary_reply()

        e = self.engine(summarize=summarize)
        new = e.before_turn(tape(4))
        self.assertNotIn("output cleared", seen[0])
        self.assertIn("r" * 2000, seen[0])
        self.assertIn("compaction", new[1])
        self.assertEqual(new[0]["text"], "the task")
        self.assertEqual(check_pairing(new), [])
        self.assertEqual(self.records[0]["method"], "summary")
        self.assertEqual(self.records[0]["count"], 1)
        self.assertTrue(e.verify_pending)
        self.assertIsNone(e.meter.anchor)

    def test_summarizer_sees_the_task_on_every_compaction(self):
        seen = []
        e = self.engine(summarize=lambda s, m: seen.append(m[0]["text"]) or summary_reply())
        first = tape(4)
        first[0]["text"] = "Build it. Constraint: standard library only."
        folded = e.compact(first, force=True)
        e.compact(folded + tape(4)[1:], force=True)
        self.assertEqual(len(seen), 2)
        for request in seen:
            self.assertIn("[User]: Build it. Constraint: standard library only.", request)
        self.assertEqual(folded[1]["compaction"]["user_messages"], [])  # pinned, not duplicated

    def test_summary_rebuilds_the_prompt(self):
        e = self.engine(summarize=lambda s, m: summary_reply())
        e.before_turn(tape(4))
        self.assertEqual(e.system, "system v2")

    def test_system_only_changes_at_compaction(self):
        # The provider's prefix cache needs a byte-identical system on ordinary turns.
        fake = ScriptedProvider([reply("a", [call("echo", "1")]), reply("b", [call("echo", "2")]),
                                 reply("done")])
        seen = []

        def complete(model, system, messages, tools):
            seen.append(system)
            return fake(model, system, messages, tools)

        e = self.engine(complete=complete, threshold=10**6)
        run_loop("main", e.system, [{"role": "user", "text": "go"}], {"echo": Echo()},
                 before_turn=e.before_turn, complete=e.complete)
        self.assertEqual(seen, ["system v1"] * 3)

    def test_nothing_to_fold_does_nothing(self):
        e = self.engine(summarize=mock.Mock(), threshold=1, keep=10**6)
        t = tape(2)
        self.assertIs(e.before_turn(t), t)
        e._summarize.assert_not_called()

    def test_redacts_summary(self):
        text = GOOD + "\nkey sk-" + "a" * 30
        e = self.engine(summarize=lambda s, m: summary_reply(text))
        new = e.before_turn(tape(4))
        self.assertNotIn("sk-aaaa", new[1]["text"])

    def test_force_skips_the_prune_stop(self):
        e = self.engine(summarize=lambda s, m: summary_reply(), threshold=4000, keep=500)
        self.assertIn("compaction", e.compact(tape(6), force=True)[1])

    def test_focus_reaches_the_request(self):
        seen = []
        e = self.engine(summarize=lambda s, m: seen.append(m[0]["text"]) or summary_reply())
        e.compact(tape(4), force=True, focus="the parser")
        self.assertIn("Pay particular attention to: the parser", seen[0])

    def test_small_threshold_shrinks_the_word_budget(self):
        seen = []
        e = self.engine(summarize=lambda s, m: seen.append(m[0]["text"]) or summary_reply())
        e.before_turn(tape(4))
        self.assertIn("Stay under 300 words", seen[0])


class FailureTests(Base):
    def test_validation_retry_names_missing_headings(self):
        replies = [summary_reply(GOOD.replace("## Goal\n", "")), summary_reply()]
        seen = []

        def summarize(system, messages):
            seen.append(messages[0]["text"])
            return replies.pop(0)

        e = self.engine(summarize=summarize)
        self.assertIn("compaction", e.before_turn(tape(4))[1])
        self.assertIn("omitted these required headings: ## Goal", seen[1])

    def assert_history_kept(self, summarize):
        e = self.engine(summarize=summarize)
        t = tape(4)
        new = e.before_turn(t)
        self.assertEqual(len(new), len(t))
        self.assertNotIn("compaction", new[1])
        self.assertEqual(e.pause_turns, 5)
        self.assertEqual(check_pairing(new), [])
        self.assertIn("keeping history", self.stderr.getvalue())
        return e

    def test_call_failure_keeps_history_and_pauses(self):
        e = self.assert_history_kept(mock.Mock(side_effect=RuntimeError("HTTP 503")))
        for left in (4, 3, 2, 1, 0):
            self.assertIs(type(e.before_turn(tape(4))), list)
            self.assertEqual(e.pause_turns, left)

    def test_max_tokens_is_a_failure(self):
        self.assert_history_kept(lambda s, m: summary_reply(finish="MAX_TOKENS"))

    def test_two_invalid_summaries_is_a_failure(self):
        self.assert_history_kept(lambda s, m: summary_reply("no headings"))

    def test_rebuild_failure_keeps_old_prompt(self):
        def build():
            self.builds += 1
            if self.builds > 1:
                raise RuntimeError("over budget")
            return "system v1"

        e = ContextEngine("main", build, [], window=100_000, threshold=1000, keep_tokens=100,
                          summarize=lambda s, m: summary_reply())
        self.assertIn("compaction", e.before_turn(tape(4))[1])
        self.assertEqual(e.system, "system v1")

    def test_summary_model_falls_back_once(self):
        calls = []

        def complete(model, system, messages, tools):
            calls.append(model)
            if model == "cheap":
                raise RuntimeError("HTTP 503")
            return summary_reply()

        e = self.engine(complete=complete, summary_model="cheap")
        self.assertIn("compaction", e.before_turn(tape(4))[1])
        self.assertEqual(calls, ["cheap", "main"])

    def test_summary_model_env_var(self):
        with mock.patch.dict("os.environ", {"ORION_COMPACT_MODEL": "cheap"}):
            self.assertEqual(self.engine().summary_model, "cheap")
        self.assertEqual(self.engine(summary_model="x").summary_model, "x")


class BreakerTests(Base):
    def test_two_ineffective_compactions_pause(self):
        complete = lambda *a: reply("x", usage=usage(5000))  # noqa: E731
        e = self.engine(summarize=lambda s, m: summary_reply(), complete=complete)
        t = e.before_turn(tape(4))
        e.complete("main", "", t, [])
        self.assertEqual((e.ineffective, e.pause_turns), (1, 0))
        t = e.before_turn(t + tape(4)[1:])
        e.complete("main", "", t, [])
        self.assertEqual(e.pause_turns, 10)
        self.assertIn("pausing it for 10 turns", self.stderr.getvalue())

    def test_effective_compaction_clears_the_strike(self):
        e = self.engine(summarize=lambda s, m: summary_reply(),
                        complete=lambda *a: reply("x", usage=usage(500)))
        e.ineffective = 1
        t = e.before_turn(tape(4))
        e.complete("main", "", t, [])
        self.assertEqual(e.ineffective, 0)


class OverflowTests(Base):
    def script(self, *outcomes):
        outcomes = list(outcomes)

        def complete(model, system, messages, tools):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return complete

    def test_recovers_once_in_place(self):
        e = self.engine(summarize=lambda s, m: summary_reply(),
                        complete=self.script(provider.ContextOverflow("too big"), reply("ok")))
        t = tape(4)
        out = e.complete("main", "", t, [])
        self.assertEqual(out["text"], "ok")
        self.assertIn("compaction", t[1])  # the loop's list, changed in place
        self.assertEqual(check_pairing(t), [])

    def test_failed_summary_uses_fallback_here(self):
        e = self.engine(summarize=mock.Mock(side_effect=RuntimeError("down")),
                        complete=self.script(provider.ContextOverflow("too big"), reply("ok")))
        t = tape(4)
        e.complete("main", "", t, [])
        self.assertTrue(t[1]["compaction"]["degraded"])
        self.assertEqual(self.records[0]["method"], "fallback")
        self.assertEqual(summary.validate(t[1]["compaction"]["summary"]), [])

    def test_second_overflow_names_sizes(self):
        e = self.engine(summarize=lambda s, m: summary_reply(),
                        complete=self.script(provider.ContextOverflow("a"),
                                             provider.ContextOverflow("b")))
        with self.assertRaises(provider.ContextOverflow) as ctx:
            e.complete("main", "", tape(4), [])
        for word in ("system+tools", "tape", "first message"):
            self.assertIn(word, str(ctx.exception))

    def test_pressure_alone_when_nothing_can_fold(self):
        t = [{"role": "user", "text": "go"}] + tape(1, 50_000)[1:]
        e = self.engine(summarize=mock.Mock(), keep=10**6,
                        complete=self.script(provider.ContextOverflow("a"), reply("ok")))
        self.assertEqual(e.complete("main", "", t, [])["text"], "ok")
        e._summarize.assert_not_called()


class AccountingTests(Base):
    def test_meter_does_not_count_the_reply_twice(self):
        fake = ScriptedProvider([reply("y" * 4000, usage=usage(1000, 1000))])
        e = self.engine(complete=fake, threshold=10**6)
        messages = [{"role": "user", "text": "go"}]
        run_loop("main", e.system, messages, {}, before_turn=e.before_turn, complete=e.complete)
        self.assertEqual(e.meter.current(messages), 2000)

    def test_summary_spend_reaches_run_result_but_not_the_anchor(self):
        fake = ScriptedProvider([reply("a", [call("echo", "1")], usage=usage(2000)),
                                 reply("done", usage=usage(300))])
        e = self.engine(complete=fake, threshold=1000, keep=50,
                        summarize=lambda s, m: summary_reply())
        messages = tape(4)
        result = run_loop("main", e.system, messages, {"echo": Echo()},
                          before_turn=e.before_turn, complete=e.complete)
        self.assertEqual(result.usage["input"], 2000 + 300 + 100 + 100)
        self.assertEqual(result.usage["output"], 20 + 40 + 40)
        self.assertEqual(result.usage["thoughts"], 6 + 5 + 5)
        self.assertEqual(result.usage["cached"], 2 + 2)
        self.assertEqual(e.meter.anchor, 310)  # raw usage, no summary spend
        self.assertEqual(check_pairing(messages), [])

    def test_long_run_stays_legal_end_to_end(self):
        replies = [reply("", [call("echo", f"t{i}")], usage=usage(200 + 900 * (i % 3)))
                   for i in range(12)] + [reply("finished", usage=usage(300))]
        fake = ScriptedProvider(replies)
        e = self.engine(complete=fake, threshold=1000, keep=120,
                        summarize=lambda s, m: summary_reply())
        messages = [{"role": "user", "text": "Constraint: standard library only."}]
        result = run_loop("main", e.system, messages, {"echo": Echo()},
                          before_turn=e.before_turn, complete=e.complete)
        self.assertEqual(result.stop, "done")
        self.assertEqual(check_pairing(messages), [])
        self.assertEqual(messages[0]["text"], "Constraint: standard library only.")
        self.assertTrue(self.records)
        for seen in fake.seen:
            self.assertEqual(check_pairing(seen), [])

    def test_overhead_matches_estimate(self):
        e = self.engine()
        self.assertEqual(e.meter.overhead, estimate_overhead("system v1", []))


if __name__ == "__main__":
    unittest.main()

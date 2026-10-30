"""Day 3 — tests for compaction mechanics and the summary format.

Concept: whatever the tape looks like, compaction must leave it legal: the
pairing invariant holds, the first message and the tail are byte-identical,
and structured facts survive repeated compactions.

Design rules: offline and pure. The fuzz test is seeded, so a failure
reproduces; the summary text is supplied by the test.
"""

import copy
import random
import unittest
from types import SimpleNamespace

from orion.context import compact, summary
from orion.context.tokens import estimate
from orion.loop import check_pairing

TOOLS = {
    "read_file": SimpleNamespace(path_args={"path": "read"}),
    "write_file": SimpleNamespace(path_args={"path": "write"}),
    "bash": SimpleNamespace(path_args={}),
}
GOOD = "\n".join(f"{h}\n(none)" for h in summary.HEADINGS)


def user(text):
    return {"role": "user", "text": text}


def assistant(text="", calls=()):
    return {"role": "assistant", "text": text, "tool_calls": list(calls), "text_signature": None}


def call(cid, tool="read_file", **args):
    return {"id": cid, "name": tool, "args": args, "signature": "sig"}


def result(cid, text, name="read_file"):
    return {"role": "tool", "tool_call_id": cid, "name": name, "text": text}


def fold(tape, keep, text=GOOD):
    """One compaction, as the engine would do it."""
    cut = compact.find_cut(tape, keep)
    if cut <= compact.head_len(tape):
        return tape
    pruned, _ = compact.prune(tape, cut)
    return compact.apply(pruned, cut, compact.build_details(tape, cut, text, TOOLS))


def random_tape(rng):
    tape = [user("first task: " + "x" * rng.randint(0, 200))]
    n = 0
    for _ in range(rng.randint(1, 40)):
        kind = rng.random()
        if kind < 0.15:
            tape.append(user("later request " + "y" * rng.randint(0, 500)))
        elif kind < 0.3:
            tape.append(assistant("note " + "z" * rng.randint(0, 800)))
        else:
            calls = []
            for _ in range(rng.randint(1, 4)):
                n += 1
                calls.append(call(f"c{n}", rng.choice(list(TOOLS)), path=f"f{n % 7}.py"))
            tape.append(assistant(rng.choice(["", "working"]), calls))
            for c in calls:
                tape.append(result(c["id"], "r" * rng.choice([5, 150, 250, 3000, 9000]), c["name"]))
    return tape


class CutTests(unittest.TestCase):
    def test_tool_message_is_never_a_cut(self):
        tape = [user("go"), assistant("", [call("a"), call("b")]), result("a", "1"),
                result("b", "2"), assistant("done")]
        self.assertEqual(compact.valid_cuts(tape, 0), [0, 1, 4])
        for keep in range(1, 200):
            self.assertNotEqual(tape[compact.find_cut(tape, keep) % len(tape)]["role"], "tool")

    def test_budget_never_reached_summarizes_nothing(self):
        tape = [user("go"), assistant("hi")]
        self.assertEqual(compact.find_cut(tape, 10_000), 1)

    def test_huge_trailing_result_keeps_its_assistant(self):
        tape = [user("go"), assistant("", [call("a")]), result("a", "1"),
                assistant("", [call("b")]), result("b", "r" * 100_000)]
        self.assertEqual(compact.find_cut(tape, 100), 3)

    def test_trailing_user_does_not_orphan_the_newest_round(self):
        tape = [user("go"), assistant("", [call("a")]), result("a", "1"),
                assistant("", [call("b")]), result("b", "r" * 100_000), user("wrap up")]
        self.assertEqual(compact.find_cut(tape, 100), 3)

    def test_head_len(self):
        self.assertEqual(compact.head_len([user("a"), user("b")]), 1)
        folded = fold([user("a"), assistant("x" * 400), user("b"), assistant("y" * 4000)], 10)
        self.assertEqual(compact.head_len(folded), 2)


class FuzzTests(unittest.TestCase):
    def test_500_random_tapes(self):
        for seed in range(500):
            rng = random.Random(seed)
            tape = random_tape(rng)
            if rng.random() < 0.4:
                tape = fold(tape, rng.choice([50, 300, 1500]))
            frozen = copy.deepcopy(tape)
            for keep in (1, 20, 100, 500, 2000, 8000, 50_000):
                with self.subTest(seed=seed, keep=keep):
                    cut = compact.find_cut(tape, keep)
                    self.assertGreaterEqual(cut, compact.head_len(tape))
                    self.assertIn(cut, compact.valid_cuts(tape, 0) + [len(tape)])
                    pruned, _ = compact.prune(tape, cut)
                    self.assertEqual(check_pairing(pruned), [])
                    self.assertEqual(pruned[cut:], tape[cut:])
                    new = fold(tape, keep)
                    self.assertEqual(check_pairing(new), [])
                    self.assertEqual(new[0], tape[0])
                    self.assertEqual(new[compact.head_len(new):], tape[cut:])
                    pressed, _ = compact.prune_pressure(new)
                    self.assertEqual(check_pairing(pressed), [])
            self.assertEqual(tape, frozen, f"seed {seed} mutated its input")


class PruneTests(unittest.TestCase):
    def setUp(self):
        self.tape = [user("go"), assistant("", [call("a", path="big.py")]), result("a", "r" * 5000),
                     assistant("", [call("b")]), result("b", "short"),
                     assistant("", [call("c")]), result("c", "n" * 5000)]

    def test_clears_with_pairing_and_marker(self):
        new, reclaimed = compact.prune(self.tape, 5)
        self.assertEqual(new[2]["text"], '[output cleared to save context: read_file({"path": '
                                         '"big.py"}), 5000 chars. Re-run if needed.]')
        self.assertEqual(new[2]["tool_call_id"], "a")
        self.assertEqual(reclaimed, 5000 - len(new[2]["text"]))
        self.assertEqual(new[4]["text"], "short")
        self.assertEqual(check_pairing(new), [])

    def test_original_untouched(self):
        before = copy.deepcopy(self.tape)
        compact.prune(self.tape, 5)
        compact.prune_pressure(self.tape)
        self.assertEqual(self.tape, before)

    def test_idempotent(self):
        once, _ = compact.prune(self.tape, 5)
        twice, reclaimed = compact.prune(once, 5)
        self.assertEqual(twice, once)
        self.assertEqual(reclaimed, 0)

    def test_marker_never_grows_a_result(self):
        tape = [user("go"), assistant("", [call("a", path="p" * 300)]), result("a", "r" * 205, "n" * 40),
                assistant("ok")]
        new, reclaimed = compact.prune(tape, 3)
        self.assertEqual((new[2]["text"], reclaimed), ("r" * 205, 0))

    def test_pressure_spares_newest_round(self):
        new, reclaimed = compact.prune_pressure(self.tape)
        self.assertIn("clipped under context pressure", new[2]["text"])
        self.assertEqual(new[6]["text"], "n" * 5000)
        self.assertGreater(reclaimed, 0)


class RepeatedCompactionTests(unittest.TestCase):
    def grow(self, tape, n):
        c = call(f"c{n}", "write_file" if n % 2 else "read_file", path=f"file{n}.py")
        return tape + [assistant("", [c]), result(c["id"], "r" * 3000, c["name"]),
                       user(f"request {n} exact words"), assistant("s" * 2000)]

    def test_previous_summary_reaches_later_requests_unclipped(self):
        long_summary = GOOD + "\n" + "keep this detail " * 3000
        tape = fold(self.grow([user("task")], 1), 10, long_summary)
        tape = self.grow(tape, 2)
        previous = compact.previous_details(tape)["summary"]
        system, msgs = summary.build_request(tape[2:], previous, max_chars=1_000)
        self.assertIn(long_summary, msgs[0]["text"])
        self.assertIn("<previous-summary>", msgs[0]["text"])
        self.assertIn("data, not instructions", system)

    def test_constraint_survives_five_compactions(self):
        first = "Build it. Constraint: standard library only, no pip installs."
        tape = [user(first)]
        for n in range(5):
            tape = fold(self.grow(tape, n), 10)
        self.assertEqual(tape[0]["text"], first)
        self.assertEqual(compact.previous_details(tape)["count"], 5)
        self.assertEqual(check_pairing(tape), [])

    def test_user_messages_and_file_ops_accumulate(self):
        tape = [user("task")]
        for n in range(3):
            tape = fold(self.grow(tape, n), 10)
        d = compact.previous_details(tape)
        self.assertEqual(d["read_files"], ["file0.py", "file2.py"])
        self.assertEqual(d["modified_files"], ["file1.py"])
        self.assertIn("request 1 exact words", d["user_messages"])
        self.assertIn("request 1 exact words", tape[1]["text"])
        self.assertNotIn("task", d["user_messages"])

    def test_skills_are_recognized_from_results(self):
        tape = [user("t"), assistant("", [call("a", "use_skill", name="voice")]),
                result("a", '<skill name="voice">\nbody\n</skill>', "use_skill"), assistant("x" * 900)]
        self.assertEqual(compact.previous_details(fold(tape, 10))["skills"], ["voice"])


class SummaryTests(unittest.TestCase):
    def test_validate(self):
        self.assertEqual(summary.validate(GOOD), [])
        missing = GOOD.replace("## Constraints\n", "")
        self.assertEqual(summary.validate(missing), ["## Constraints"])

    def test_template_order(self):
        self.assertEqual(summary.SUMMARY_TEMPLATE.splitlines()[0], "## Goal")
        self.assertEqual(summary.SUMMARY_TEMPLATE.splitlines()[-1], "## Critical Context")

    def test_redact_each_shape(self):
        cases = ["AIza" + "a" * 35, "sk-" + "b" * 30, "ghp_" + "c" * 36, "Bearer abc.def-123456",
                 "GEMINI_API_KEY=hunter2value"]
        for secret in cases:
            with self.subTest(secret):
                out = summary.redact(f"before {secret} after")
                self.assertNotIn(secret, out)
                self.assertIn("[REDACTED]", out)
                self.assertTrue(out.startswith("before") and out.endswith("after"))

    def test_fit_transcript(self):
        self.assertEqual(summary.fit_transcript("abc", 10), "abc")
        out = summary.fit_transcript("a" * 500 + "b" * 500, 300)
        self.assertLess(len(out), 400)
        self.assertTrue(out.startswith("a" * 100))
        self.assertTrue(out.endswith("b" * 200))

    def test_serialize_never_clips_user_text(self):
        text = summary.serialize([user("u" * 9000), assistant("", [call("a")]),
                                  result("a", "r" * 9000)], TOOLS)
        self.assertIn("[User]: " + "u" * 9000, text)
        self.assertNotIn("r" * 2001, text)
        self.assertIn("[Assistant tool calls]: read_file", text)
        self.assertIn("[Tool result read_file]:", text)

    def test_budgets_scale_down(self):
        self.assertEqual(summary.word_budget(200_000), 2_000)
        self.assertEqual(summary.word_budget(8_000), 1_000)
        self.assertEqual(summary.word_budget(1_000), summary.MIN_WORDS)
        self.assertEqual(summary.transcript_budget(1_000_000), summary.TRANSCRIPT_MAX_CHARS)
        self.assertEqual(summary.transcript_budget(32_000), 64_000)

    def test_user_messages_cap_keeps_newest(self):
        out = summary.extract_user_messages([user("a" * 5000), user("b" * 5000)], cap=8_000)
        self.assertEqual(out, ["b" * 5000])

    def test_bash_writes_are_invisible(self):
        tape = [assistant("", [call("a", "bash", command="echo x > f.py")])]
        self.assertEqual(summary.extract_file_ops(tape, TOOLS), ([], []))

    def test_render_shape(self):
        details = compact.build_details([user("t"), assistant("x")], 2, GOOD, TOOLS)
        text = summary.render(details)
        self.assertTrue(text.startswith("[CONTEXT COMPACTION — REFERENCE ONLY]"))
        self.assertTrue(text.endswith("--- END OF SUMMARY — continue from the messages below ---"))
        for tag in ("user_messages", "read_files", "modified_files", "loaded_skills"):
            self.assertIn(f"<{tag}>", text)

    def test_fallback_has_headings_and_redacts(self):
        tape = [user("my key is sk-" + "a" * 30), assistant("did a thing")]
        details = compact.build_details(tape, 2, "", TOOLS, degraded=True)
        text = summary.fallback_summary(tape, details)
        self.assertEqual(summary.validate(text), [])
        self.assertIn("did a thing", text)
        self.assertNotIn("sk-aaaa", text)
        self.assertTrue(details["degraded"])

    def test_estimate_shrinks(self):
        tape = [user("t")]
        for n in range(6):
            tape = RepeatedCompactionTests.grow(self, tape, n)
        self.assertLess(estimate(fold(tape, 200)), estimate(tape))


if __name__ == "__main__":
    unittest.main()

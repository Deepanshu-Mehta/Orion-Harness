"""Day 3 — the context engine: when to shrink the tape, and how far.

Concept: compaction escalates, cheapest first, and stops as soon as the tape
is small enough. Prune clears old tool output for free; summarize folds the
middle for the price of one model call. The engine plugs into the loop's two
existing sockets, `before_turn` and `complete`, so the loop never changes.

Design rules:
- Never drop history because a summary failed. Keep the pruned tape, pause,
  and try again later; only a real overflow, where the history cannot fit
  anyway, may use the model-free fallback.
- A tail too big to fold must not buy a paid summary every turn: two
  ineffective compactions in a row pause compaction.
- Real usage decides. The meter anchors on what the provider reports, and
  the system prompt changes only at compaction, so the provider's cached
  prefix survives every ordinary turn.
- Summary spend is reported in the turn's usage, so totals and token budgets
  include it.
"""

import math
import os
import sys

from orion import provider
from orion.context import compact, summary
from orion.context.tokens import (CHARS_PER_TOKEN, TokenMeter, compaction_threshold,
                                  estimate, estimate_overhead, keep_recent)

PRUNE_ENOUGH = 0.7      # prune alone is enough below this share of the threshold
FAILURE_PAUSE = 5
THRASH_PAUSE = 10
THRASH_STRIKES = 2
_USAGE_KEYS = ("input", "output", "thoughts", "cached")


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


class ContextEngine:
    """Keeps the tape inside the window; wire in `before_turn` and `complete`."""

    def __init__(self, model, system_builder, tool_specs, window=None, threshold=None,
                 keep_tokens=None, summarize=None, summary_model=None, on_compaction=None,
                 tools=None, complete=None):
        """Bind to a model.

        `system_builder()` returns the fitted system prompt. `summarize(system,
        messages)` returns a provider-shaped reply; by default it is a model
        call. `complete` is the main model call, injectable like the loop's.
        """
        self.model = model
        self.system_builder = system_builder
        self.tool_specs = tool_specs
        self.tools = tools or {}
        self.on_compaction = on_compaction
        self._complete = complete or provider.complete
        self._summarize = summarize or self._summary_call
        self.summary_model = summary_model or os.environ.get("ORION_COMPACT_MODEL") or model
        self.window = window or provider.context_window(model)
        self.threshold = threshold or compaction_threshold(self.window)
        self.keep_tokens = keep_tokens or keep_recent(self.threshold)
        self.system = system_builder()
        self.meter = TokenMeter(self._overhead())
        self.pause_turns = 0
        self.verify_pending = False
        self.ineffective = 0
        self._pending = dict.fromkeys(_USAGE_KEYS, 0)

    def _overhead(self) -> int:
        return estimate_overhead(self.system, self.tool_specs)

    # -- the before_turn socket -----------------------------------------------

    def before_turn(self, messages: list) -> list:
        """Return the tape to send next, compacted if it has grown too large."""
        if self.pause_turns > 0:
            self.pause_turns -= 1
            return messages
        if self.meter.current(messages) < self.threshold:
            return messages
        return self.compact(messages)

    def compact(self, messages: list, force: bool = False, focus: str | None = None) -> list:
        """Prune, then summarize if pruning was not enough. `force` skips the early stop."""
        before = self.meter.current(messages)
        cut = compact.find_cut(messages, self.keep_tokens)
        if cut <= compact.head_len(messages):
            return messages
        pruned, reclaimed = compact.prune(messages, cut)
        # Measure from the real anchor minus what was cleared: re-estimating the
        # whole tape would throw away the provider's number.
        after = before - math.ceil(reclaimed / CHARS_PER_TOKEN)
        if not force and reclaimed and after < PRUNE_ENOUGH * self.threshold:
            self.meter.reset(self._overhead())  # the anchor described the unpruned tape
            self._report(messages, pruned, before, cut, reclaimed, "prune", "")
            return pruned
        # Summarize the unpruned span: cleared outputs would tell the summary nothing.
        text = self._make_summary(messages, cut, focus)
        if text is None:
            _warn(f"summary failed; keeping history and pausing compaction for "
                  f"{FAILURE_PAUSE} turns")
            self.pause_turns = FAILURE_PAUSE
            if reclaimed:
                self.meter.reset(self._overhead())
            return pruned
        return self._fold(messages, pruned, cut, text, before, reclaimed, degraded=False)

    def _fold(self, messages, tape, cut, text, before, reclaimed, degraded) -> list:
        details = compact.build_details(messages, cut, text, self.tools, degraded)
        new = compact.apply(tape, cut, details)
        self._rebuild_system()
        self.meter.reset(self._overhead())
        self.verify_pending = True
        self._report(messages, new, before, cut, reclaimed,
                     "fallback" if degraded else "summary", text, details["count"], degraded)
        return new

    def _rebuild_system(self) -> None:
        """Refresh the prompt at compaction, when the cached prefix is gone anyway."""
        try:
            self.system = self.system_builder()
        except RuntimeError as e:  # e.g. memory grew past the budget: never crash a run here
            _warn(f"could not rebuild the system prompt, keeping the old one: {e}")

    def _report(self, old, new, before, cut, reclaimed, method, text, count=None,
                degraded=False) -> None:
        if not self.on_compaction:
            return
        previous = compact.previous_details(old) or {}
        self.on_compaction({
            "count": previous.get("count", 0) if count is None else count,
            "tokens_before": before,
            "tokens_after_estimate": self._overhead() + estimate(new),
            "cut_at": cut,
            "pruned_chars": reclaimed,
            "summary_chars": len(text),
            "degraded": degraded,
            "method": method,
            "summary": text,
        })

    # -- summary calls ----------------------------------------------------------

    def _summary_call(self, system: str, messages: list) -> dict:
        """The default summarizer: the summary model, falling back once to the main one."""
        try:
            return self._complete(self.summary_model, system, messages, [])
        except RuntimeError:
            if self.summary_model == self.model:
                raise
            _warn(f"summary model {self.summary_model} failed; using {self.model}")
            return self._complete(self.model, system, messages, [])

    def _make_summary(self, messages: list, cut: int, focus: str | None) -> str | None:
        """Return validated, redacted summary text, or None if the call failed."""
        previous = (compact.previous_details(messages) or {}).get("summary")
        # The pinned first message is not folded, but it is the task: without it the
        # summarizer cannot state the Goal or quote the Constraints.
        span = messages[:1] + messages[compact.head_len(messages):cut]
        system, request = summary.build_request(
            span, previous, focus, self.tools, summary.transcript_budget(self.window),
            summary.word_budget(self.threshold))
        for attempt in range(2):
            try:
                reply = self._summarize(system, request)
            except RuntimeError as e:
                _warn(f"summary call failed: {e}")
                return None
            self._add_pending(reply.get("usage") or {})
            if reply.get("finish") == "MAX_TOKENS":
                return None
            missing = summary.validate(reply["text"])
            if not missing:
                return summary.redact(reply["text"])
            names = ", ".join(missing)
            request = [{"role": "user", "text": f"{request[0]['text']}\n\nYour last attempt "
                        f"omitted these required headings: {names}. Include every heading."}]
        return None

    def _add_pending(self, usage: dict) -> None:
        for k in _USAGE_KEYS:
            self._pending[k] += usage.get(k, 0)

    # -- the complete socket ------------------------------------------------------

    def complete(self, model, system, messages, tools) -> dict:
        """Model call for the loop; recovers from overflow and accounts for compaction."""
        # The loop's `system` is from the start of the run; ours follows compaction.
        try:
            reply = self._complete(model, self.system, messages, tools)
        except provider.ContextOverflow:
            messages[:] = self._recover(messages)
            try:
                reply = self._complete(model, self.system, messages, tools)
            except provider.ContextOverflow as e:
                raise provider.ContextOverflow(
                    f"{e} | still over after compaction: system+tools ~{self._overhead()} "
                    f"tokens, tape ~{estimate(messages)} tokens in {len(messages)} messages, "
                    f"first message ~{estimate(messages[:1])} tokens") from None
        # The reply is appended right after this returns, and its size is already
        # in usage["output"]: +1 keeps it from being estimated a second time.
        self.meter.record(reply["usage"], len(messages) + 1)
        self._check_effect(reply["usage"])
        return self._with_pending(reply)

    def _check_effect(self, usage: dict) -> None:
        """The anti-thrash breaker: did the last compaction really get under?"""
        if not self.verify_pending:
            return
        self.verify_pending = False
        if usage["input"] + usage["output"] < self.threshold:
            self.ineffective = 0
            return
        self.ineffective += 1
        if self.ineffective >= THRASH_STRIKES:
            _warn(f"compaction is not getting under {self.threshold} tokens; pausing it for "
                  f"{THRASH_PAUSE} turns")
            self.pause_turns = THRASH_PAUSE
            self.ineffective = 0

    def _recover(self, messages: list) -> list:
        """One forced compaction after an overflow; here a failed summary may not stop it."""
        before = self.meter.current(messages)
        pressed, reclaimed = compact.prune_pressure(messages)
        cut = compact.find_cut(pressed, self.keep_tokens)
        if cut <= compact.head_len(pressed):
            self.meter.reset(self._overhead())
            return pressed
        pruned, more = compact.prune(pressed, cut)
        reclaimed += more
        text = self._make_summary(messages, cut, None)
        degraded = text is None
        if degraded:
            details = compact.build_details(messages, cut, "", self.tools, True)
            text = summary.fallback_summary(messages, details)
        return self._fold(messages, pruned, cut, text, before, reclaimed, degraded)

    def _with_pending(self, reply: dict) -> dict:
        """Add summary spend to the reply's usage, after the meter saw the raw numbers."""
        if not any(self._pending.values()):
            return reply
        usage = {k: reply["usage"].get(k, 0) + self._pending[k] for k in _USAGE_KEYS}
        self._pending = dict.fromkeys(_USAGE_KEYS, 0)
        return {**reply, "usage": usage}

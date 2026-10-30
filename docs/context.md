# Orion context model

One question shapes this layer: **how does a long task fit inside a finite
window?** There are three kinds of content, and each gets its own mechanism.

```
                     SYSTEM PROMPT (assembled, never compacted, capped)
  stable ─► identity + rules ─► environment ─► project instructions
  volatile ─────────────────► skills catalog ─► memory ─► date
                     ─────────────────────────────────────
                     TAPE (grows, managed by the context engine)
  [first user message, verbatim]
  [compaction summary: structured, updated in place]    ◄─ the summarized middle
  [recent tail: last ~20k tokens, verbatim]             ◄─ starts on a legal message
```

## Measurement

The provider reports the true size of each prompt, so the last report is an
anchor and only what came after it is estimated (`orion/context/tokens.py`).
Thoughts are billed but not re-sent, so they stay out. Compaction starts at
`min(0.75 × (window − 16,384), 200,000)` tokens. A 1M window still compacts at
200k because every prompt token is paid again on every turn, and because Pro
input pricing doubles above 200k.

## The escalation

Each step runs only if the one before was not enough.

1. **Tool caps** (Day 2) stop growth at the source.
2. **Prune** replaces old tool outputs with a one-line marker. No model call.
   If that brings the prompt under 70% of the threshold, the turn goes ahead.
3. **Summarize** folds the middle into a structured summary. One model call,
   made on the original outputs, not the pruned ones.

If a summary call fails, the pruned tape is kept and compaction pauses for 5
turns: history is never dropped because a summary failed. If two compactions
in a row do not get under the threshold, it pauses for 10, so a tail too big to
fold cannot buy a paid summary every turn. On a real `ContextOverflow`, the
engine forces one compaction, clipping large results in the tail too, and may
use a model-free fallback summary, because that history cannot fit anyway.

## Prompt caching

Orion does not use explicit context caching. It relies on Gemini's implicit
caching, which needs no code and no storage fee: on `gemini-3.1-pro-preview`,
once a prompt reaches 4,096 tokens, a repeated prefix is billed at a tenth of
the input price. Explicit caches cost storage per token-hour, cannot follow a
tape that grows every turn, and are not supported from Gemini 4 Argon on.

The caching work is keeping the prefix stable:

- The system prompt runs stable to volatile and holds the date, never a clock.
- It is built once and rebuilt only at compaction, when the prefix has already
  changed, so `remember` notes take effect then.
- The tape only appends between compactions (`tests/test_prompt.py` proves the
  wire format stays a prefix; `tests/test_engine.py` proves the system string
  does not change on ordinary turns).
- Prune and summarize both end the cached prefix for one turn. That is why
  neither runs before the threshold.

The per-turn line in `demos/day3_context.py` prints the cached share, which
shows whether this is working. DeepSeek caches prefixes the same way, on by
default: a 14-turn live run with no compaction served 90% of its input tokens
from cache, and runs with compaction served 79%.

## Compared with pi and Hermes

Orion takes pi's structured, updated-in-place summary and Hermes's protected
head and cheap prune step:

- **From pi**: fixed headings, an update prompt that carries Constraints and
  Preferences word for word, and structured details stored on the message, so
  a later compaction reads fields rather than parsing prose.
- **From Hermes**: the first user message stays pinned verbatim, and old tool
  output is cleared before any money is spent.
- **Different**: a `tool` message is never a cut point, so the tape needs no
  repair step: the illegal state cannot be constructed. The summary runs on
  unpruned output and always sees the pinned first message, which is the task.
  Summary word and transcript budgets scale with the window.

## Instructions, memory and skills

| | Project instructions | Agent memory | Skills |
|---|---|---|---|
| File | `AGENTS.md` / `CLAUDE.md` | `.orion/memory.md` | `.orion/skills/<name>/SKILL.md` |
| Written by | humans | the agent, via `remember` | humans |
| In the prompt | in full, capped | in full, labeled facts, not instructions | name and description; full text on `use_skill` |

All three can arrive with a cloned repository, so all three are checked: text
bound for the prompt cannot resolve to a protected file (`AGENTS.md -> .env`
is refused), and project content that fails `scan_injection` is not loaded.

If the prompt and tools take more than half of what the reply leaves,
`fit_prompt` degrades them in a fixed order: skill descriptions, the skills
catalog, memory, project instructions, web tool descriptions. If that is not
enough it raises `ContextBudgetError` before any model call.

## Known limits

- **`bash` file writes aren't tracked.** `read_files` and `modified_files` come
  from `Tool.path_args`, so a file changed by a shell command is not listed.
- **The token estimate is rough.** Four characters per token, used only for the
  few messages since the last anchor and for tapes just after a compaction.
- **Injection scanning is a speed bump.** Regexes leak; rephrasing beats them,
  and a legitimate file that discusses injection can be blocked. Isolation is
  the boundary.
- **The first user message is pinned.** A huge first message cannot be folded;
  a persistent overflow reports its size.
- **Thought signature rules are not fully documented.** The tail is kept
  verbatim, signatures included. Live runs on `gemini-3-flash-preview` accepted
  compacted tapes, but the API's exact rules for trimmed history are not
  published.
- **Prune discards detail.** A cleared output must be re-run if needed again.
- **DeepSeek's reasoning is re-read but not in the anchor.** It is stored in
  `text_signature` and echoed each turn, as the API requires when tools are in
  use. The anchor counts it as thoughts, so the meter runs one reply low until
  the next real usage arrives.
- **Tiny thresholds can stay over.** If the prompt, the summary and the tail do
  not fit under the threshold, the breaker pauses compaction. This shows up in
  demos with a small `--compact-headroom`, not at real window sizes.

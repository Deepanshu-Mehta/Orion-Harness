"""Day 3 — live demo: a long task inside a finite window.

Concept: Day 2's agent, now with an assembled system prompt (instructions,
skills, memory), and a context engine that shrinks the tape as it grows.
Everything enters through the loop's existing sockets; the loop is unchanged.

Design rules:
- The budget preflight runs before any model call, so an impossible prompt
  fails loudly without spending money.
- `--compact-headroom N` sets the threshold to overhead + N: a fixed tiny
  threshold cannot work, because the system prompt and schemas alone are
  larger, so the demo prints the overhead.
- The per-turn line shows cache hits, because Gemini's implicit prefix
  caching is the cheapest optimization here and only measurement proves the
  prompt stayed stable.
"""

import argparse
import dataclasses
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orion import provider  # noqa: E402
from orion.context.engine import ContextEngine  # noqa: E402
from orion.context.tokens import estimate_overhead  # noqa: E402
from orion.env import make_env  # noqa: E402
from orion.loop import run_loop  # noqa: E402
from orion.memory import remember_tool  # noqa: E402
from orion.prompt import ContextBudgetError, build_system_prompt, fit_prompt  # noqa: E402
from orion.security import Gate, Policy, toolset_for  # noqa: E402
from orion.skills import skill_tool  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TTY = sys.stdout.isatty()


def _color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def _clip(text, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def approve(call, decision) -> str:
    """Ask the human on the terminal; returns "yes", "no" or "always"."""
    print(_color(f"  ? {decision.reason}", "33"))
    try:
        answer = input(f"approve {call['name']}({_clip(json.dumps(call['args']), 120)})? "
                       "[y/N/a] ").strip().lower()
    except EOFError:
        answer = ""
    return {"y": "yes", "yes": "yes", "a": "always", "always": "always"}.get(answer, "no")


def print_decision(call, decision):
    """Print gate decisions; allows by the mode table stay quiet."""
    if decision.action == "allow" and decision.layer in ("mode", "unknown"):
        return
    line = f"  gate: {decision.action} [{decision.layer}] {_clip(decision.reason, 200)}"
    print(_color(line, "31") if decision.action == "deny" else _color(line, "2"))


def event_printer(engine, messages):
    """Build the on_event callback; it reads the engine's live numbers."""
    def on_event(kind, payload):
        if kind == "assistant" and payload["text"]:
            print(f"assistant: {payload['text']}")
        elif kind == "tool_start":
            print(f"  -> {payload['name']}({_clip(json.dumps(payload['args']), 160)})")
        elif kind == "tool_end":
            line = f"  <- {_clip(payload['result'], 300)}"
            print(_color(line, "31") if payload["result"].startswith("BLOCKED") else line)
        elif kind == "turn_end":
            u = payload["usage"]
            hit = 100 * u["cached"] // u["input"] if u["input"] else 0
            print(_color(f"[turn {payload['turn']}] in={u['input']} out={u['output']} "
                         f"cached={u['cached']} ({hit}%) | context: "
                         f"{engine.meter.current(messages)} / {engine.threshold}", "2"))
    return on_event


def compaction_printer(show_summary):
    """Build the on_compaction callback."""
    def on_compaction(r):
        print(_color(f"compaction #{r['count']}: {r['tokens_before']} → "
                     f"{r['tokens_after_estimate']} ({r['method']})", "36"))
        if show_summary and r["summary"]:
            print(_color(r["summary"], "2"))
    return on_compaction


def parse_args(argv):
    """Command-line options for the demo."""
    p = argparse.ArgumentParser(description="Run Orion on a task with context management.")
    p.add_argument("task", nargs="*", help="the task (quoted or as several words)")
    p.add_argument("--env", default="local",
                   help="local | sandbox | docker[:image] | ssh:<host>:<abs_root>")
    p.add_argument("--mode", default="ask", help="read-only | ask | accept-edits | auto (yolo)")
    p.add_argument("--web", action="store_true", help="enable web_search and web_fetch")
    p.add_argument("--exec-network", action="store_true",
                   help="let commands reach the network (sandbox and docker)")
    p.add_argument("--workdir", help="workspace root (default: a fresh dir under ./scratch/)")
    p.add_argument("--headless", action="store_true", help="no approver: asks are denied")
    p.add_argument("--trace", action="store_true", help="print the full message tape at the end")
    p.add_argument("--max-turns", type=int, default=80)
    p.add_argument("--compact-headroom", type=int, metavar="N",
                   help="compact at overhead + N tokens (for demos on small tasks)")
    p.add_argument("--show-summary", action="store_true", help="print each summary as it lands")
    args = p.parse_args(argv)
    if not args.task:
        p.error("give a task, e.g. \"Create fib.py and run it\"")
    return args


def main(argv) -> int:
    """Build env, gate, tools, prompt and engine; run the task; always close the env."""
    args = parse_args(argv)
    workdir = args.workdir
    if workdir is None:
        os.makedirs(os.path.join(REPO, "scratch"), exist_ok=True)
        workdir = tempfile.mkdtemp(prefix="run-", dir=os.path.join(REPO, "scratch"))
    try:
        env = make_env(args.env, workdir, exec_network=args.exec_network)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        model = provider.default_model()
        policy = Policy.load(env.root, env=env)
        extra = {"remember": remember_tool(env), "use_skill": skill_tool(env)}
        tools = toolset_for(args.mode, env, allow_web=args.web, policy=policy, extra_tools=extra)
        gate = Gate(args.mode, env, tools, approver=None if args.headless else approve,
                    policy=policy, allow_web=args.web, on_decision=print_decision)
        window = provider.context_window(model)
        network = tuple(n for n, t in tools.items() if t.kind == "network")

        # Preflight: compaction cannot shrink the prompt, so fail before spending.
        system, specs, warnings = fit_prompt(build_system_prompt(env, gate),
                                             [t.spec for t in tools.values()], window, network)
        for warning in warnings:
            print(_color(f"warning: {warning}", "33"))
        by_name = {s["schema"]["name"]: s for s in specs}
        tools = {n: dataclasses.replace(t, spec=by_name[n]) for n, t in tools.items()}

        overhead = estimate_overhead(system, specs)
        threshold = overhead + args.compact_headroom if args.compact_headroom else None
        engine = ContextEngine(
            model, lambda: fit_prompt(build_system_prompt(env, gate), specs, window, network)[0],
            specs, window=window, threshold=threshold, tools=tools,
            on_compaction=compaction_printer(args.show_summary))

        print(_color(f"env: {env.describe()}", "2"))
        print(_color(f"mode: {gate.mode} | tools: {', '.join(tools)}", "2"))
        print(_color(f"model: {model} | window: {window} | overhead: ~{overhead} | "
                     f"threshold: {engine.threshold} | keep: {engine.keep_tokens}", "2"))
        task = " ".join(args.task)
        print(f"user: {task}")
        messages = [{"role": "user", "text": task}]
        result = run_loop(model, engine.system, messages, tools,
                          on_event=event_printer(engine, messages), before_tool=gate.check,
                          before_turn=engine.before_turn, complete=engine.complete,
                          max_turns=args.max_turns)
        if args.trace:
            print(json.dumps(messages, indent=2))
        print(f"\nfinal: {result.text}")
        print(f"stop={result.stop} turns={result.turns} usage={result.usage}")
        u = result.usage
        if u["input"]:
            print(f"cache: {u['cached']} of {u['input']} input tokens "
                  f"({100 * u['cached'] // u['input']}%)")
        return 0
    except (ContextBudgetError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Day 2 — live demo: an agent with hands, behind a gate, inside a box.

Concept: wire the five rings together. Build an environment (where tools
run), a gate (whether they run), and the toolset the mode allows; then run
the Day 1 loop unchanged, plugging the gate into `before_tool`.

Design rules:
- The loop is untouched; everything enters through `tools` and `before_tool`.
- Interactive by default: an "ask" prompts the human. `--headless` has no
  approver, so asks fail closed.
- The environment is always closed (jobs stopped, containers removed), even
  when the run fails.
"""

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orion import provider  # noqa: E402
from orion.env import make_env  # noqa: E402
from orion.loop import run_loop  # noqa: E402
from orion.security import Gate, Policy, environment_prompt, toolset_for  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDENTITY = ("You are Orion, a coding agent working in the workspace described below. "
            "Act, don't narrate; verify what you build by running it.")

_TTY = sys.stdout.isatty()


def _color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def approve(call, decision) -> str:
    """Ask the human on the terminal; returns "yes", "no" or "always"."""
    args = _clip(json.dumps(call["args"]), 120)
    print(_color(f"  ? {decision.reason}", "33"))
    try:
        answer = input(f"approve {call['name']}({args})? [y/N/a] ").strip().lower()
    except EOFError:
        answer = ""
    return {"y": "yes", "yes": "yes", "a": "always", "always": "always"}.get(answer, "no")


def print_event(kind, payload):
    """Print loop events compactly."""
    if kind == "assistant" and payload["text"]:
        print(f"assistant: {payload['text']}")
    elif kind == "tool_start":
        print(f"  -> {payload['name']}({_clip(json.dumps(payload['args']), 160)})")
    elif kind == "tool_end":
        result = payload["result"]
        line = f"  <- {_clip(result, 300)}"
        print(_color(line, "31") if result.startswith("BLOCKED") else line)
    elif kind == "turn_end":
        u, t = payload["usage"], payload["total"]
        print(_color(f"[turn {payload['turn']}] in={u['input']} out={u['output']} "
                     f"| total in={t['input']} out={t['output']}", "2"))


def print_decision(call, decision):
    """Print gate decisions; allows by the mode table stay quiet."""
    if decision.action == "allow" and decision.layer in ("mode", "unknown"):
        return
    line = f"  gate: {decision.action} [{decision.layer}] {_clip(decision.reason, 200)}"
    print(_color(line, "31") if decision.action == "deny" else _color(line, "2"))


def parse_args(argv):
    """Command-line options for the demo."""
    p = argparse.ArgumentParser(description="Run Orion on a task in a chosen environment.")
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
    p.add_argument("--max-turns", type=int, default=40)
    args = p.parse_args(argv)
    if not args.task:
        p.error("give a task, e.g. \"Create fib.py and run it\"")
    return args


def main(argv) -> int:
    """Build env, gate and toolset; run the task; always close the env."""
    args = parse_args(argv)
    task = " ".join(args.task)
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
        policy = Policy.load(env.root, env=env)
        tools = toolset_for(args.mode, env, allow_web=args.web, policy=policy)
        gate = Gate(args.mode, env, tools, approver=None if args.headless else approve,
                    policy=policy, allow_web=args.web, on_decision=print_decision)
        system = f"{IDENTITY}\n\n{environment_prompt(env, gate)}"
        print(_color(f"env: {env.describe()}", "2"))
        print(_color(f"mode: {gate.mode} | tools: {', '.join(tools)}", "2"))
        print(f"user: {task}")
        messages = [{"role": "user", "text": task}]
        result = run_loop(provider.default_model(), system, messages, tools,
                          on_event=print_event, before_tool=gate.check,
                          max_turns=args.max_turns)
        if args.trace:
            print(json.dumps(messages, indent=2))
        print(f"\nfinal: {result.text}")
        print(f"stop={result.stop} turns={result.turns} usage={result.usage}")
        return 0
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

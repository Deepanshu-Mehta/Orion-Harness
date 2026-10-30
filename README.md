# Orion-Harness

Orion is a minimal coding-agent harness written in pure Python (standard library only, Python 3.10+), built in five layers: the loop, tools, context, durability, and orchestration. Day 1 provides the model client (`orion/provider.py`, the only file that knows the vendor) and the agent loop (`orion/loop.py`). Day 2 adds hands and limits: environments that decide where tools run (`orion/env/`), typed tools (`orion/tools/`), and a permission gate (`orion/security/`). Day 3 fits long tasks in a finite window: an assembled system prompt (`orion/prompt.py`), memory and skills (`orion/memory.py`, `orion/skills.py`), and a context engine that prunes and summarizes the tape (`orion/context/`). Run the offline tests with `python3 -m unittest discover -s tests -v` (no API key or network needed; Docker and SSH tests are skipped unless enabled with `ORION_TEST_DOCKER=1` or `ORION_TEST_SSH=host:/abs/path`).

## Demos

Set `ORION_API_KEY` or `GEMINI_API_KEY` (and optionally `ORION_MODEL`) first.

```sh
# Day 1: one tool, one loop
python3 demos/day1_dice.py --trace

# Day 2: an agent with tools, behind the gate, in an OS sandbox
python3 demos/day2_build.py --env sandbox --mode auto --headless \
  "Create fib.py with an iterative fib(n) and a __main__ that prints fib(30); run it"

# Interactive approvals on your machine (the default mode is "ask")
python3 demos/day2_build.py --env local "Add a README to this project"

# Read-only tour of a repository
python3 demos/day2_build.py --env local --mode read-only --workdir . "Explain what this repository does"

# Web tools are opt-in
python3 demos/day2_build.py --env sandbox --mode auto --headless --web \
  "What is the latest stable Python version? Cite the source."

# Day 3: compaction on a small task; --compact-headroom N compacts at overhead + N tokens
python3 demos/day3_context.py --env sandbox --mode auto --headless --compact-headroom 2500 --show-summary \
  "Create five files one.txt through five.txt, then MANIFEST.md listing each file's line count"
```

`--env` accepts `local`, `sandbox`, `docker`, `docker:<image>`, or `ssh:<host>:<abs_root>`. `--mode` accepts `read-only`, `ask`, `accept-edits`, or `auto` (alias `yolo`). Without `--workdir`, each run gets a fresh directory under `./scratch/`.

## Context model

The system prompt is assembled once, ordered stable to volatile, and checked against the window before any model call. Project instructions come from `AGENTS.md` or `CLAUDE.md`, skills from `.orion/skills/<name>/SKILL.md`, and memory from `.orion/memory.md`. When the tape grows too large, old tool output is pruned first, and only then is the middle summarized. Orion relies on Gemini's implicit prefix caching rather than explicit caches. See [docs/context.md](docs/context.md).

## Security model

Every tool call passes five rings: schema validation, the gate (hard-deny command rules, protected paths, the mode dial, and a human approver), the path jail, the operations layer, and OS isolation. The gate decides whether a call runs; the environment decides how far a mistake can reach. `auto` mode refuses to run without isolation. See [docs/security.md](docs/security.md) for the mode table, what each environment does and does not protect against, and the known leaks of pattern rules.

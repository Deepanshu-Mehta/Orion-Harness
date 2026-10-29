"""Day 2 — security package: the gate that decides whether a tool call runs.

Concept: modes, protected paths, command rules and policy files combine
into one Decision per call; the environment then bounds what an allowed
call can reach.

Design rules: see gate.py (decision order), rules.py (command rules),
paths.py (protected paths) and policy.py (tighten-only config).
"""

from orion.security.gate import (MODES, Decision, Gate, environment_prompt, is_isolated,
                                 normalize_mode, toolset_for)
from orion.security.policy import Policy

__all__ = ["MODES", "Decision", "Gate", "Policy", "environment_prompt", "is_isolated",
           "normalize_mode", "toolset_for"]

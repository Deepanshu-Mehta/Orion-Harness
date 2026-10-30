"""Day 3 — the context package: how a long task fits inside a finite window.

Concept: the system prompt is assembled and capped, the tape is measured
with the provider's real numbers, and compaction shrinks the tape in
escalating steps: prune first, summarize only when pruning is not enough.

Design rules: see tokens.py (measurement), compact.py (safe cut points),
summary.py (the checkpoint format) and engine.py (the policy).
"""

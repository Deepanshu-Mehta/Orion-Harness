"""Day 3 — trust checks for text bound for the system prompt.

Concept: instructions, memory and skills reload every session and sit in the
most trusted part of the prompt, and a cloned repository can ship all of
them. Everything read for the prompt therefore passes two checks: it must
not be a protected file, and it must not look like an injection.

Design rules:
- `read_text` is the one reader for workspace files bound for the prompt: a
  symlink such as `AGENTS.md -> .env` would otherwise put secrets in the
  prompt without any tool call for the gate to see.
- `scan_injection` is a speed bump, not a boundary. Regexes leak; isolation
  is what bounds the damage. Quantifiers are bounded so a hostile file
  cannot make the scan itself slow.
"""

import re
import sys

from orion.security import paths, rules

_INVISIBLE = re.compile("[​‌‎‏‪-‮⁠-⁤⁦-⁩]")
_PATTERNS = [
    ("instruction override", re.compile(
        r"(?i)\b(?:ignore|disregard|forget|override)\b[^.\n]{0,30}"
        r"\b(?:previous|prior|above|earlier|preceding)\b[^.\n]{0,20}"
        r"\b(?:instructions?|prompts?|rules?|directions?)\b")),
    ("hiding from the user", re.compile(
        r"(?i)\b(?:do not|don't|never)\s+(?:tell|inform|mention|reveal|show|let)\b[^.\n]{0,30}"
        r"\b(?:the\s+)?(?:user|human|owner)\b"
        r"|\b(?:hide|conceal|keep)\b[^.\n]{0,30}\bfrom\s+the\s+(?:user|human|owner)\b")),
    ("secret exfiltration", re.compile(
        r"(?i)\b(?:curl|wget|nc)\b[^\n]{0,120}"
        r"(?:\$\{?[A-Z_]*(?:KEY|TOKEN|SECRET|PASSWORD)|\bprintenv\b|\.env\b|\.ssh/|id_rsa|credentials)"
        r"|\b(?:env|printenv)\b[^\n]{0,40}\|\s*(?:curl|wget|nc)\b"
        r"|\bcat\b[^\n|]{0,80}(?:\.env\b|\.ssh/|id_rsa|credentials)[^\n]{0,40}\|\s*(?:curl|wget|nc)\b")),
]
_warned = set()


def warn(message: str) -> None:
    """Print a warning once; prompts are rebuilt at every compaction."""
    if message not in _warned:
        _warned.add(message)
        print(f"warning: {message}", file=sys.stderr)


def scan_injection(text: str) -> list[str]:
    """Names of the injection patterns found in `text`."""
    findings = []
    if _INVISIBLE.search(text):
        findings.append("invisible Unicode")
    findings += [name for name, pattern in _PATTERNS if pattern.search(text)]
    return findings


def read_text(env, rel: str, policy=None) -> str | None:
    """Read a workspace file for the prompt; None if missing or refused."""
    deny = policy.deny if policy is not None else paths.DEFAULT_DENY
    try:
        resolved = env.resolve(rel)
        if paths.match(resolved, deny, env.root, rules.default_home()):
            warn(f"{rel} resolves to a protected file; not loading it")
            return None
        if not env.exists(resolved) or env.is_dir(resolved):
            return None
        return env.read_bytes(resolved).decode("utf-8", "replace")
    except PermissionError:
        warn(f"{rel} resolves outside the workspace; not loading it")
        return None

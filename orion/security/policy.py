"""Day 2 — policy files: configuration that can only tighten.

Concept: defaults, then the user's `~/.orion/policy.json`, then the
project's `<root>/.orion/policy.json`, merged so each layer can add rules
but none can remove a deny.

Design rules:
- Merging only appends. `!` exceptions in files are ignored (they would
  loosen a deny), and recorded so the user can see they had no effect.
- `allow_commands` loosens the gate, so it is honored only from the user's
  own file. A project file arrives with whatever repository was cloned and
  is untrusted.
- A malformed policy fails loudly instead of being skipped: a typo must not
  silently drop a rule.
- The agent cannot edit either file: `.orion/**` is deny_write, `~/.orion`
  is deny, and the sandbox profile blocks both.
"""

import json
import os
import re
from dataclasses import dataclass, field

from orion.security import paths

LIST_KEYS = ("deny", "deny_write", "ask_write", "deny_commands", "ask_commands",
             "allow_commands")


@dataclass
class Policy:
    """The merged rule set the gate enforces."""

    deny: list = field(default_factory=lambda: list(paths.DEFAULT_DENY))
    deny_write: list = field(default_factory=lambda: list(paths.DEFAULT_DENY_WRITE))
    ask_write: list = field(default_factory=lambda: list(paths.DEFAULT_ASK_WRITE))
    deny_commands: list = field(default_factory=list)   # regexes
    ask_commands: list = field(default_factory=list)    # regexes
    allow_commands: list = field(default_factory=list)  # read-only command prefixes
    sources: list = field(default_factory=list)
    ignored: list = field(default_factory=list)

    @classmethod
    def load(cls, root: str, env=None, home: str | None = None) -> "Policy":
        """Merge defaults, the user policy, and the project policy.

        The project file is read through `env` when given (so remote
        workspaces work), else from the local filesystem.
        """
        policy = cls()
        home = home or os.path.expanduser("~")
        user_file = os.path.join(home, ".orion", "policy.json")
        if os.path.isfile(user_file):
            with open(user_file, encoding="utf-8") as f:
                policy.merge(f.read(), user_file, trusted=True)
        project_file = os.path.join(root, ".orion", "policy.json")
        if env is not None:
            if env.exists(".orion/policy.json"):
                policy.merge(env.read_bytes(".orion/policy.json").decode("utf-8"),
                             project_file, trusted=False)
        elif os.path.isfile(project_file):
            with open(project_file, encoding="utf-8") as f:
                policy.merge(f.read(), project_file, trusted=False)
        return policy

    def merge(self, text: str, source: str, trusted: bool) -> None:
        """Append one policy file's rules; never removes anything."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"policy file {source} is not valid JSON: {e}") from None
        if not isinstance(data, dict):
            raise ValueError(f"policy file {source} must contain a JSON object")
        unknown = set(data) - set(LIST_KEYS)
        if unknown:
            raise ValueError(f"policy file {source} has unknown keys: {', '.join(sorted(unknown))}"
                             f"; valid keys: {', '.join(LIST_KEYS)}")
        for key, values in data.items():
            if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
                raise ValueError(f"policy file {source}: {key} must be a list of strings")
            if key == "allow_commands" and not trusted:
                self.ignored += [f"{source}: allow_commands {v!r} (project files cannot "
                                 "loosen the policy)" for v in values]
                continue
            for value in values:
                if value.startswith("!"):
                    self.ignored.append(f"{source}: {key} {value!r} (exceptions cannot be added)")
                    continue
                if key.endswith("_commands") and key != "allow_commands":
                    try:
                        re.compile(value)
                    except re.error as e:
                        raise ValueError(f"policy file {source}: bad regex {value!r} in "
                                         f"{key}: {e}") from None
                target = getattr(self, key)
                if value not in target:
                    target.append(value)
        self.sources.append(source)

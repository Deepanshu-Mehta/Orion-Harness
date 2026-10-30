"""Day 3 — skills: instructions loaded only when the task needs them.

Concept: the prompt carries a cheap catalog of names and descriptions; the
model pays for a skill's full text only when it calls `use_skill`.

Design rules:
- Skills live in `~/.orion/skills/` and `.orion/skills/`, never a top-level
  folder: `.orion/` is deny-write, so the agent cannot rewrite them.
- The project wins a name clash. The home copy is read from the host, the
  workspace copy through `env`, so remote workspaces work.
- Project skills arrive with whatever repository was cloned, so they are
  scanned for injection; a hit is skipped. User skills only warn.
- A skill with no description is skipped: the catalog is what the model
  uses to decide.
"""

import os

from orion.security import rules
from orion.security.trust import read_text, scan_injection, warn
from orion.tools.base import Tool, ToolError, tool

SKILLS_DIR = ".orion/skills"
SKILL_FILE = "SKILL.md"
SKILL_MAX_CHARS = 50_000
DESCRIPTION_CLIP = 200


def parse_front_matter(text: str) -> dict:
    """Read `key: value` lines between leading `---` fences; values may continue indented."""
    lines = text.lstrip("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fields, key = {}, None
    for line in lines[1:]:
        if line.strip() == "---":
            return fields
        if line[:1].isspace() and key:
            fields[key] = f"{fields[key]} {line.strip()}".strip()
        elif ":" in line:
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            fields[key] = "" if value in (">", "|", ">-", "|-") else value.strip("\"'")
    return {}  # no closing fence


def _user_skills() -> dict:
    base = os.path.join(rules.default_home(), SKILLS_DIR)
    found = {}
    if os.path.isdir(base):
        for name in sorted(os.listdir(base)):
            path = os.path.join(base, name, SKILL_FILE)
            if os.path.isfile(path):
                with open(path, encoding="utf-8", errors="replace") as f:
                    found[name] = (path, f.read())
    return found


def _project_skills(env) -> dict:
    found = {}
    for name in env.list_dir(SKILLS_DIR):
        path = f"{SKILLS_DIR}/{name}/{SKILL_FILE}"
        text = read_text(env, path)
        if text is not None:
            found[name] = (path, text)
    return found


def catalog(env) -> dict:
    """name -> {description, path, scope}; project skills shadow user skills."""
    skills = {}
    for scope, found in (("user", _user_skills()), ("project", _project_skills(env))):
        for dirname, (path, text) in found.items():
            meta = parse_front_matter(text)
            name, description = meta.get("name") or dirname, meta.get("description", "")
            if not description:
                warn(f"skill {name} ({path}) has no description; skipped")
                continue
            findings = scan_injection(text)
            if findings:
                warn(f"skill {name} ({path}) looks like prompt injection: {', '.join(findings)}"
                     + ("; skipped" if scope == "project" else ""))
                if scope == "project":
                    continue
            skills[name] = {"description": description, "path": path, "scope": scope}
    return dict(sorted(skills.items()))


def catalog_prompt(env, max_chars: int = 4_000) -> str:
    """The `<available_skills>` block; "" when there are none."""
    skills = catalog(env)
    if not skills:
        return ""
    lines, used = [], 0
    for name, info in skills.items():
        line = f"- {name}: {info['description'][:DESCRIPTION_CLIP]}"
        if used + len(line) > max_chars:
            line = f"- {name}"  # over the cap: the name still lets the model ask for it
        lines.append(line)
        used += len(line) + 1
    body = "\n".join(lines)
    return (f"<available_skills>\nCall use_skill before doing work one of these covers.\n"
            f"{body}\n</available_skills>")


def skill_tool(env) -> Tool:
    """Build `use_skill`, which reads the catalog fresh on every call."""

    @tool("Load the full instructions of a skill from the available skills list.",
          kind="read", name="Skill name exactly as listed")
    def use_skill(name: str) -> str:
        skills = catalog(env)
        if name not in skills:
            known = ", ".join(skills) or "none"
            raise ToolError(f"no skill {name!r}; available skills: {known}")
        info = skills[name]
        if info["scope"] == "project":
            text = read_text(env, info["path"])
            if text is None:
                raise ToolError(f"skill {name!r} could not be read; tell the user")
        else:
            with open(info["path"], encoding="utf-8", errors="replace") as f:
                text = f.read()
        if len(text) > SKILL_MAX_CHARS:
            text = text[:SKILL_MAX_CHARS] + f"\n[...skill clipped at {SKILL_MAX_CHARS} chars]"
        return f'<skill name="{name}">\n{text}\n</skill>'

    return use_skill

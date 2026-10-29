"""Day 2 — tools package: typed contracts, the core toolset, opt-in web tools.

Concept: what the agent can touch, expressed as a small set of validated
verbs over an Environment.

Design rules: tools reach the outside world only through an Environment
(core) or through guarded HTTP (web); the gate decides which are offered.
"""

from orion.tools.base import Tool, ToolError, tool
from orion.tools.core import core_tools
from orion.tools.web import web_tools

__all__ = ["Tool", "ToolError", "core_tools", "tool", "web_tools"]

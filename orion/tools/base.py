"""Day 2 — typed tool contracts.

Concept: a tool is a Python function whose type hints become the JSON schema
the model sees, and whose wrapper validates the model's arguments before
any code runs. Bad arguments never reach the function.

Design rules:
- Schema from type hints: str, int, float, bool, list[str], Literal[...],
  and list[TypedDict] for structured items. Parameters with defaults are
  optional.
- The wrapper rejects unknown arguments, names missing ones, and coerces
  only lossless forms ("3" -> 3, "true" -> True).
- Every error string starts with `ERROR:` and says what to do next, because
  the model reads it and has to recover on its own.
- `kind` and `path_args` are metadata for the gate, never sent to the model.
"""

import inspect
import typing
from dataclasses import dataclass, field
from typing import Callable, Literal, get_args, get_origin, get_type_hints


class ToolError(Exception):
    """An expected failure; the wrapper returns it as `ERROR: <message>`."""


@dataclass
class Tool:
    """A tool as the loop and the gate see it."""

    name: str
    spec: dict                 # {"schema": {...}}, what the provider ships
    run: Callable[..., str]    # validated entry point used by the loop
    kind: str                  # "read" | "write" | "exec" | "network"
    path_args: dict = field(default_factory=dict)  # arg name -> "read" | "write"


_SCALARS = {str: "string", int: "integer", float: "number", bool: "boolean"}
_TYPE_WORDS = {str: "a string", int: "a whole number", float: "a number",
               bool: "true or false"}


def _schema_for(hint) -> dict:
    if hint in _SCALARS:
        return {"type": _SCALARS[hint]}
    if get_origin(hint) is Literal:
        return {"type": "string", "enum": [str(v) for v in get_args(hint)]}
    if get_origin(hint) is list:
        (item,) = get_args(hint)
        if typing.is_typeddict(item):
            fields = get_type_hints(item)
            return {"type": "array", "items": {
                "type": "object",
                "properties": {k: _schema_for(t) for k, t in fields.items()},
                "required": list(fields),
            }}
        return {"type": "array", "items": _schema_for(item)}
    raise TypeError(f"unsupported tool parameter type: {hint!r}")


def _coerce(name: str, value, hint):
    """Return the value converted to `hint`, or raise ToolError naming the fix."""
    if hint is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
    elif hint is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                pass
    elif hint is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                pass
    elif hint is str:
        if isinstance(value, str):
            return value
    elif get_origin(hint) is Literal:
        choices = [str(v) for v in get_args(hint)]
        if str(value) in choices:
            return str(value)
        raise ToolError(f"argument '{name}' must be one of {', '.join(choices)}; got {value!r}")
    elif get_origin(hint) is list:
        if not isinstance(value, list):
            raise ToolError(f"argument '{name}' must be a list; got {type(value).__name__}")
        (item,) = get_args(hint)
        if typing.is_typeddict(item):
            return [_coerce_object(f"{name}[{i}]", v, item) for i, v in enumerate(value)]
        return [_coerce(f"{name}[{i}]", v, item) for i, v in enumerate(value)]
    if hint in _TYPE_WORDS:
        raise ToolError(f"argument '{name}' must be {_TYPE_WORDS[hint]}; got {value!r}")
    raise TypeError(f"unsupported tool parameter type: {hint!r}")


def _coerce_object(name: str, value, typed_dict) -> dict:
    fields = get_type_hints(typed_dict)
    if not isinstance(value, dict):
        raise ToolError(f"{name} must be an object with keys {', '.join(fields)}")
    extra = set(value) - set(fields)
    missing = [k for k in fields if k not in value]
    if extra or missing:
        raise ToolError(f"{name} must have exactly the keys {', '.join(fields)}"
                        + (f"; missing {', '.join(missing)}" if missing else "")
                        + (f"; unexpected {', '.join(sorted(extra))}" if extra else ""))
    return {k: _coerce(f"{name}.{k}", value[k], t) for k, t in fields.items()}


def _os_error(e: OSError) -> str:
    target = e.filename or ""
    if isinstance(e, FileNotFoundError):
        return f"ERROR: no such file or directory: {target}; check the path with list_files"
    if isinstance(e, IsADirectoryError):
        return f"ERROR: {target} is a directory; use list_files to see what is inside"
    if isinstance(e, NotADirectoryError):
        return f"ERROR: a parent of {target} is a file, not a directory; choose another path"
    if isinstance(e, PermissionError):
        return f"ERROR: {e}; use a path inside the workspace that you are allowed to access"
    return f"ERROR: {type(e).__name__}: {e}"


def tool(description: str, kind: str, paths: dict | None = None, **param_descriptions):
    """Decorator: build a Tool from a function's signature and type hints.

    `paths` maps argument names to "read" or "write" so the gate can check
    them; `param_descriptions` documents each parameter for the model.
    """
    def decorate(fn) -> Tool:
        hints = get_type_hints(fn)
        params = inspect.signature(fn).parameters
        properties, required, defaults = {}, [], {}
        for pname, p in params.items():
            prop = _schema_for(hints.get(pname, str))
            if pname in param_descriptions:
                prop["description"] = param_descriptions[pname]
            properties[pname] = prop
            if p.default is inspect.Parameter.empty:
                required.append(pname)
            else:
                defaults[pname] = p.default
        schema = {"name": fn.__name__, "description": description,
                  "parameters": {"type": "object", "properties": properties}}
        if required:
            schema["parameters"]["required"] = required

        def run(**args) -> str:
            unknown = [a for a in args if a not in properties]
            if unknown:
                return (f"ERROR: unknown argument '{unknown[0]}'; valid: "
                        f"{', '.join(properties)}")
            missing = [a for a in required if a not in args]
            if missing:
                return (f"ERROR: missing required argument(s) {', '.join(missing)}; "
                        f"call {fn.__name__} again with them set")
            try:
                values = {k: _coerce(k, v, hints.get(k, str)) for k, v in args.items()}
                return fn(**values)
            except ToolError as e:
                return f"ERROR: {e}"
            except OSError as e:
                return _os_error(e)

        run.__doc__ = description
        return Tool(fn.__name__, {"schema": schema}, run, kind, dict(paths or {}))
    return decorate

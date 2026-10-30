"""Day 1 — the provider: the one file that knows the vendor.

Concept: everything outside this module speaks a neutral message format
(user / assistant / tool messages, ToolCalls with ids). This module
translates that format to Gemini's wire format and back, and owns the HTTP
call with its retry policy. Day 2 adds `web_search`, a grounded call used
by the opt-in web tool, so the vendor stays confined to this file. Day 3
adds `ContextOverflow` and real model limits for the context engine.

Design rules:
- Standard library only; the API key travels in a header, never in a URL,
  and is never printed or logged.
- `_to_wire` and `_parse` are pure (no network) so tests can call them.
- Thought signatures are stored on the neutral messages and echoed back
  verbatim; Gemini 3 rejects follow-up turns that drop them.
- Every tool call gets an id, even though Gemini does not need one, because
  the pairing invariant (and any future provider) does.
"""

import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
import uuid

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3.1-pro-preview"

FALLBACK_WINDOW = 200_000

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_OVERFLOW = re.compile(r"(?i)token count.*exceeds|exceeds the maximum number of tokens"
                       r"|context.*(length|window)")
_limits_cache = {}
_warned_fallback = False


class ContextOverflow(RuntimeError):
    """The prompt was larger than the model's input limit (an HTTP 400)."""


def api_key() -> str:
    """Return the API key from ORION_API_KEY, falling back to GEMINI_API_KEY."""
    key = os.environ.get("ORION_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "No API key found: set ORION_API_KEY or GEMINI_API_KEY in the environment."
        )
    return key


def default_model() -> str:
    """Return the model name from ORION_MODEL, else DEFAULT_MODEL."""
    return os.environ.get("ORION_MODEL") or DEFAULT_MODEL


def model_limits(model: str) -> dict:
    """Return {"input", "output"} token limits for `model`, cached per model."""
    if model not in _limits_cache:
        info = _request(f"{API_ROOT}/{model}", retries=2)
        _limits_cache[model] = {"input": info["inputTokenLimit"],
                                "output": info["outputTokenLimit"]}
    return _limits_cache[model]


def context_window(model: str) -> int:
    """Input tokens the model accepts: ORION_CONTEXT_WINDOW, else the API, else a fallback.

    The fallback is deliberately modest: a wrong guess must fail small, by
    compacting early, not large, by overflowing.
    """
    global _warned_fallback
    override = os.environ.get("ORION_CONTEXT_WINDOW")
    if override:
        try:
            return int(override)
        except ValueError:
            raise ValueError(f"ORION_CONTEXT_WINDOW must be a whole number, got {override!r}") from None
    try:
        return model_limits(model)["input"]
    except (RuntimeError, KeyError, ValueError) as e:
        if not _warned_fallback:
            _warned_fallback = True
            print(f"warning: could not read limits for {model} ({e}); "
                  f"assuming a {FALLBACK_WINDOW}-token window", file=sys.stderr)
        return FALLBACK_WINDOW


def complete(model: str, system: str, messages: list, tools: list) -> dict:
    """Make one model call and return text, tool calls, finish reason and usage.

    `messages` is in the neutral format; `tools` is a list of
    `{"schema": {...}}` dicts (possibly empty).
    """
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": _to_wire(messages),
        # No temperature: Gemini 3 recommends its default; lower values can loop.
        "generationConfig": {"maxOutputTokens": 65536},
    }
    if tools:
        body["tools"] = [{"functionDeclarations": [t["schema"] for t in tools]}]
    return _parse(_post(f"{API_ROOT}/{model}:generateContent", body))


def web_search(query: str, model: str | None = None) -> dict:
    """Answer `query` with a Google-Search-grounded call.

    Returns {"text": str, "sources": [{"title", "url"}]}. This is a separate
    generateContent call with the google_search tool, not part of the loop.
    """
    body = {
        "contents": [{"role": "user", "parts": [{"text": query}]}],
        "tools": [{"google_search": {}}],
    }
    url = f"{API_ROOT}/{model or default_model()}:generateContent"
    return _parse_grounding(_post(url, body))


def _parse_grounding(response: dict) -> dict:
    """Extract answer text and web sources from a grounded response."""
    text = _parse(response)["text"]
    meta = response["candidates"][0].get("groundingMetadata") or {}
    sources = [{"title": c["web"].get("title", ""), "url": c["web"].get("uri", "")}
               for c in meta.get("groundingChunks") or [] if "web" in c]
    return {"text": text, "sources": sources}


def _to_wire(messages: list) -> list:
    """Convert neutral messages to Gemini `contents`, alternating user/model."""
    contents = []
    for m in messages:
        if m["role"] == "user":
            content = {"role": "user", "parts": [{"text": m["text"]}]}
        elif m["role"] == "assistant":
            parts = []
            if m.get("text"):
                part = {"text": m["text"]}
                if m.get("text_signature"):
                    part["thoughtSignature"] = m["text_signature"]
                parts.append(part)
            for call in m.get("tool_calls") or []:
                part = {"functionCall": {"name": call["name"], "args": call["args"]}}
                # Signature echo: Gemini 3 returns HTTP 400 if this is dropped.
                if call.get("signature"):
                    part["thoughtSignature"] = call["signature"]
                parts.append(part)
            content = {"role": "model", "parts": parts}
        elif m["role"] == "tool":
            response = {"name": m["name"], "response": {"result": m["text"]}}
            content = {"role": "user", "parts": [{"functionResponse": response}]}
        else:
            raise ValueError(f"unknown message role: {m['role']!r}")

        # Result grouping: consecutive tool messages (and any same-role
        # neighbours) merge into one content, keeping call order, so the
        # wire strictly alternates user/model.
        if contents and contents[-1]["role"] == content["role"]:
            contents[-1]["parts"].extend(content["parts"])
        else:
            contents.append(content)
    return contents


def _parse(response: dict) -> dict:
    """Convert a Gemini generateContent response into the neutral result dict."""
    candidates = response.get("candidates") or []
    if not candidates:
        raise RuntimeError(
            f"model returned no candidates; promptFeedback={response.get('promptFeedback')}"
        )
    candidate = candidates[0]
    texts, tool_calls, text_signature = [], [], None
    for part in (candidate.get("content") or {}).get("parts") or []:
        if "functionCall" in part:
            fc = part["functionCall"]
            tool_calls.append({
                "id": fc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                "name": fc["name"],
                "args": fc.get("args") or {},
                "signature": part.get("thoughtSignature"),
            })
        elif "text" in part and not part.get("thought"):
            texts.append(part["text"])
            if part.get("thoughtSignature"):
                text_signature = part["thoughtSignature"]

    meta = response.get("usageMetadata") or {}
    return {
        "text": "".join(texts),
        "tool_calls": tool_calls,
        "text_signature": text_signature,
        "finish": candidate.get("finishReason", ""),
        "usage": {
            "input": meta.get("promptTokenCount", 0),
            "output": meta.get("candidatesTokenCount", 0),
            "thoughts": meta.get("thoughtsTokenCount", 0),
            "cached": meta.get("cachedContentTokenCount", 0),
        },
    }


def _post(url: str, body: dict, retries: int = 5) -> dict:
    """POST JSON to `url`, retrying rate limits, server errors and network faults."""
    return _request(url, json.dumps(body).encode("utf-8"), retries)


def _request(url: str, data: bytes | None = None, retries: int = 5) -> dict:
    """Send one API request (GET when `data` is None) under the retry policy."""
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key()}
    last_failure = "no attempts made"
    for attempt in range(retries):
        retry_after = None
        try:
            method = "GET" if data is None else "POST"
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code not in _RETRYABLE_STATUS:
                detail = e.read().decode("utf-8", "replace")
                kind = ContextOverflow if e.code == 400 and _OVERFLOW.search(detail) else RuntimeError
                raise kind(f"HTTP {e.code} from model API: {detail[:400]}") from e
            last_failure = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"
            retry_after = e.headers.get("Retry-After")
        except (urllib.error.URLError, TimeoutError) as e:
            last_failure = f"{type(e).__name__}: {e}"
        if attempt < retries - 1:
            time.sleep(_retry_delay(attempt, retry_after))
    raise RuntimeError(f"model API failed after {retries} attempts; last failure: {last_failure}")


def _retry_delay(attempt: int, retry_after) -> float:
    """Seconds to wait: the server's Retry-After if numeric, else jittered backoff."""
    if retry_after is not None:
        try:
            return float(retry_after)
        except ValueError:
            pass  # HTTP-date form; fall back to backoff
    return 2 ** attempt * 2 + random.random()

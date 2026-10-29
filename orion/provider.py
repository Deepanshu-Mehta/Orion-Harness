"""Day 1 — the provider: the one file that knows the vendor.

Concept: everything outside this module speaks a neutral message format
(user / assistant / tool messages, ToolCalls with ids). This module
translates that format to Gemini's wire format and back, and owns the HTTP
call with its retry policy.

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
import time
import urllib.error
import urllib.request
import uuid

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3.1-pro-preview"

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


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
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key()}
    last_failure = "no attempts made"
    for attempt in range(retries):
        retry_after = None
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code not in _RETRYABLE_STATUS:
                detail = e.read().decode("utf-8", "replace")[:400]
                raise RuntimeError(f"HTTP {e.code} from model API: {detail}") from e
            last_failure = f"HTTP {e.code}"
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

"""Day 2 — opt-in web tools: search and fetch, with the web treated as hostile.

Concept: the web is where untrusted instructions come from. These tools
exist (rather than `curl` in bash) because they carry safety the shell
cannot: an SSRF guard, size and time caps, HTML reduced to text, and every
result wrapped as untrusted content.

Design rules:
- Off unless the run passes allow_web; kind "network" for the gate.
- web_fetch refuses non-public addresses on every redirect hop, so a public
  URL cannot bounce the harness into localhost or a cloud metadata service.
  Known gap: the host is resolved for the check and again for the connect,
  so DNS rebinding between the two is not prevented (see docs/security.md).
- Search returns titles, URLs and snippets only, never whole pages.
- The vendor-grounded search lives in provider.py, the only file that
  knows the vendor.
"""

import ipaddress
import json
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from orion import provider
from orion.tools.base import Tool, ToolError, tool

FETCH_TIMEOUT = 15
FETCH_MAX_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 5
USER_AGENT = "Orion/0.2 (+https://github.com/Deepanshu-Mehta/Orion-Harness)"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"


def wrap_untrusted(source: str, text: str) -> str:
    """Wrap fetched content so the model treats it as data, not instructions."""
    # A page could contain the closing tag to break out of the wrapper.
    text = text.replace("</untrusted_content", "&lt;/untrusted_content")
    return f'<untrusted_content source="{source}">\n{text}\n</untrusted_content>'


def check_public_url(url: str) -> None:
    """Raise ToolError unless url is http(s) and its host resolves only to public IPs."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise ToolError(f"refusing {url!r}: only http and https URLs can be fetched")
    host = parsed.hostname
    if not host:
        raise ToolError(f"refusing {url!r}: no host in URL")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ToolError(f"cannot resolve {host}: {e}; check the URL") from None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.version == 6 and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        # is_global excludes loopback, private, link-local (incl. the
        # 169.254.169.254 metadata address), CGNAT, and reserved ranges.
        if not ip.is_global or ip.is_multicast:
            raise ToolError(f"refusing {url!r}: {host} resolves to non-public address {ip}; "
                            "only public internet addresses can be fetched")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Surface redirects as HTTPError so each hop is checked before following."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _TextExtractor(HTMLParser):
    """Reduce HTML to readable text, dropping script, style and navigation."""

    SKIP = {"script", "style", "nav", "noscript", "svg", "template", "head"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
             "section", "article", "header", "footer", "pre", "blockquote", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip_depth, self.title = [], 0, ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in self.SKIP:
            self.skip_depth += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in self.SKIP and self.skip_depth:
            self.skip_depth -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in raw.split("\n"))
        body = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
        return f"# {self.title.strip()}\n\n{body}" if self.title.strip() else body


def html_to_text(html: str) -> str:
    """Convert an HTML document to readable plain text."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


def _fetch(url: str) -> tuple[str, str, bytes, bool]:
    """GET url following checked redirects; return (final url, content type, body, cut)."""
    opener = urllib.request.build_opener(_NoRedirect)
    for _ in range(MAX_REDIRECTS + 1):
        check_public_url(url)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with opener.open(req, timeout=FETCH_TIMEOUT) as resp:
                body = resp.read(FETCH_MAX_BYTES + 1)
                return (url, resp.headers.get("Content-Type", ""), body[:FETCH_MAX_BYTES],
                        len(body) > FETCH_MAX_BYTES)
        except urllib.error.HTTPError as e:
            location = e.headers.get("Location")
            if e.code in (301, 302, 303, 307, 308) and location:
                url = urllib.parse.urljoin(url, location)
                continue
            raise ToolError(f"HTTP {e.code} fetching {url}; try a different URL") from None
        except (urllib.error.URLError, TimeoutError) as e:
            reason = getattr(e, "reason", e)
            raise ToolError(f"could not fetch {url}: {reason}") from None
    raise ToolError(f"more than {MAX_REDIRECTS} redirects from the original URL; giving up")


def search_backend() -> str:
    """Which search backend runs: "brave" with BRAVE_API_KEY, else "gemini"."""
    return "brave" if os.environ.get("BRAVE_API_KEY") else "gemini"


def _brave_search(query: str, max_results: int) -> list[dict]:
    params = urllib.parse.urlencode({"q": query, "count": max_results})
    req = urllib.request.Request(f"{BRAVE_URL}?{params}", headers={
        "Accept": "application/json", "X-Subscription-Token": os.environ["BRAVE_API_KEY"]})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [{"title": r.get("title", ""), "url": r.get("url", ""),
             "snippet": re.sub(r"<[^>]+>", "", r.get("description", ""))}
            for r in (data.get("web") or {}).get("results", [])[:max_results]]


def web_tools() -> dict[str, Tool]:
    """Build the opt-in web tools."""

    @tool("Search the web. Returns titles, URLs and short snippets; use web_fetch to read "
          "a page.", kind="network", query="Search query", max_results="1-10 results")
    def web_search(query: str, max_results: int = 5) -> str:
        if not 1 <= max_results <= 10:
            raise ToolError("max_results must be between 1 and 10")
        if search_backend() == "brave":
            results = _brave_search(query, max_results)
            lines = [f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet'][:300]}"
                     for i, r in enumerate(results, 1)]
            return wrap_untrusted("brave-search", "\n".join(lines) or "no results")
        answer = provider.web_search(query)
        lines = [answer["text"][:1500].strip(), "", "Sources:"]
        lines += [f"{i}. {s['title']} — {s['url']}"
                  for i, s in enumerate(answer["sources"][:max_results], 1)]
        return wrap_untrusted("gemini-grounded-search", "\n".join(lines))

    @tool("Fetch a public web page and return it as readable text.", kind="network",
          url="http or https URL", max_chars="Maximum characters of text to return")
    def web_fetch(url: str, max_chars: int = 20000) -> str:
        final_url, ctype, body, cut = _fetch(url)
        charset = re.search(r"charset=([\w-]+)", ctype)
        text = body.decode(charset.group(1) if charset else "utf-8", "replace")
        if "html" in ctype or text.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
            text = html_to_text(text)
        notes = []
        if cut:
            notes.append(f"[download capped at {FETCH_MAX_BYTES // 1024 // 1024} MB]")
        if len(text) > max_chars:
            notes.append(f"[showing {max_chars} of {len(text)} chars]")
            text = text[:max_chars]
        return wrap_untrusted(final_url, "\n".join([text] + notes))

    return {web_search.name: web_search, web_fetch.name: web_fetch}

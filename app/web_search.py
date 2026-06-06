"""Lightweight DuckDuckGo web search used as a fallback when the model is unsure.

Uses the keyless HTML endpoint (https://html.duckduckgo.com/html/) and parses the
result blocks with regex so no extra dependency (bs4/lxml) is required.
"""

import html as _html
import logging
import re
import urllib.parse
from dataclasses import dataclass

import requests

LOGGER = logging.getLogger(__name__)

DDG_HTML_URL = "https://html.duckduckgo.com/html/"
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0 Safari/537.36"
)

_RESULT_RE = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_SNIPPET_RE = re.compile(
    r'class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


def _clean(fragment: str) -> str:
    return _html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def _real_url(href: str) -> str:
    # DuckDuckGo wraps links as //duckduckgo.com/l/?uddg=<urlencoded>&...
    if "uddg=" in href:
        normalized = href if href.startswith("http") else f"https:{href}"
        params = urllib.parse.parse_qs(urllib.parse.urlparse(normalized).query)
        if params.get("uddg"):
            return params["uddg"][0]
    if href.startswith("//"):
        return f"https:{href}"
    return href


def search_duckduckgo(query: str, *, max_results: int = 5, timeout: float = 8.0) -> list[SearchResult]:
    query = query.strip()
    if not query:
        return []
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.post(
            DDG_HTML_URL,
            data={"q": query},
            headers={"User-Agent": _USER_AGENT},
            timeout=timeout,
        )
        response.raise_for_status()
    except Exception:
        LOGGER.debug("duckduckgo request failed", exc_info=True)
        return []

    body = response.text
    snippets = [m.group("snippet") for m in _SNIPPET_RE.finditer(body)]
    results: list[SearchResult] = []
    for index, match in enumerate(_RESULT_RE.finditer(body)):
        if len(results) >= max_results:
            break
        url = _real_url(match.group("href"))
        title = _clean(match.group("title"))
        snippet = _clean(snippets[index]) if index < len(snippets) else ""
        if title and url.startswith("http"):
            results.append(SearchResult(title=title, url=url, snippet=snippet))
    return results


def format_results_for_prompt(results: list[SearchResult], *, max_chars: int = 2200) -> str:
    blocks = [
        f"[{index}] {result.title}\n{result.snippet}\nURL: {result.url}"
        for index, result in enumerate(results, start=1)
    ]
    return "\n\n".join(blocks)[:max_chars]


def format_results_markdown(results: list[SearchResult], *, limit: int = 5) -> str:
    if not results:
        return ""
    lines = ["Источники (DuckDuckGo):"]
    lines.extend(f"- [{result.title}]({result.url})" for result in results[:limit])
    return "\n".join(lines)

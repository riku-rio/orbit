from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlencode, urlparse

from orbit.browser.chrome_devtools import (
    ChromeDevToolsClient,
    ChromeDevToolsError,
    ChromeToolResult,
)

DEFAULT_RESULT_LIMIT = 5
GOOGLE_HOME_URL = "https://www.google.com/"
GOOGLE_SEARCH_URL = "https://www.google.com/search"
_EVALUATE_JSON_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
_PAGE_LINE_RE = re.compile(r"^(\d+):\s+(.+?)(?:\s+\[selected\])?$")

GOOGLE_RESULTS_EXTRACTOR = r"""async () => {
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
  const blockedReason = () => {
    const host = window.location.hostname.toLowerCase();
    const path = window.location.pathname.toLowerCase();
    const body = (document.body?.innerText || '').toLowerCase();
    if (host === 'consent.google.com' || host.endsWith('.consent.google.com')) {
      return 'Google presented a consent page.';
    }
    if (path.startsWith('/sorry/') || body.includes('unusual traffic from your computer network')) {
      return 'Google presented an anti-bot or unusual-traffic page.';
    }
    return null;
  };

  for (let attempt = 0; attempt < 25; attempt += 1) {
    const blocked = blockedReason();
    if (blocked) {
      return {status: 'blocked', reason: blocked, results: []};
    }
    if (document.querySelector('#search h3')) {
      break;
    }
    await sleep(200);
  }

  const blocked = blockedReason();
  if (blocked) {
    return {status: 'blocked', reason: blocked, results: []};
  }

  const host = window.location.hostname.toLowerCase();
  if (!(host === 'google.com' || host.endsWith('.google.com'))) {
    return {
      status: 'unexpected_page',
      reason: `Expected Google search results but landed on ${window.location.href}`,
      results: [],
    };
  }

  const results = [];
  const seen = new Set();
  for (const anchor of document.querySelectorAll('#search a[href]')) {
    const heading = anchor.querySelector('h3');
    if (!heading) {
      continue;
    }

    let parsed;
    try {
      parsed = new URL(anchor.href);
    } catch {
      continue;
    }
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      continue;
    }

    const resultHost = parsed.hostname.toLowerCase();
    if (resultHost === 'google.com' || resultHost.endsWith('.google.com')) {
      continue;
    }
    parsed.hash = '';
    const key = parsed.href;
    if (seen.has(key)) {
      continue;
    }

    const container =
      anchor.closest('[data-text-ad]') ||
      anchor.closest('.MjjYud') ||
      anchor.closest('.g') ||
      anchor.parentElement?.parentElement?.parentElement ||
      anchor.parentElement;
    if (anchor.closest('[data-text-ad]')) {
      continue;
    }

    const containerText = (container?.innerText || '').trim();
    const firstLines = containerText
      .split('\n')
      .map(line => line.trim())
      .filter(Boolean)
      .slice(0, 3)
      .join(' ')
      .toLowerCase();
    if (/^(sponsored|ad|advertisement)\b/.test(firstLines)) {
      continue;
    }

    const title = (heading.innerText || heading.textContent || '').trim();
    if (!title) {
      continue;
    }

    let snippet = null;
    const snippetElement = container?.querySelector(
      '.VwiC3b, .IsZvec, [data-sncf], [data-content-feature="1"]'
    );
    if (snippetElement) {
      snippet = (snippetElement.innerText || snippetElement.textContent || '').trim() || null;
    }
    if (!snippet && containerText) {
      const fallback = containerText
        .split('\n')
        .map(line => line.trim())
        .filter(line => line.length >= 40 && line !== title && !line.includes(parsed.hostname))
        .sort((a, b) => b.length - a.length)[0];
      snippet = fallback ? fallback.slice(0, 400) : null;
    }

    seen.add(key);
    results.push({
      title,
      url: parsed.href,
      source: resultHost.replace(/^www\./, ''),
      snippet,
    });
    if (results.length >= 5) {
      break;
    }
  }

  return {status: 'ok', results};
}"""


class WebSearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class BrowserPage:
    id: int
    url: str
    selected: bool = False


@dataclass(frozen=True)
class SearchResult:
    rank: int
    title: str
    source: str
    url: str
    snippet: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "snippet": self.snippet,
        }


@dataclass(frozen=True)
class WebSearchResponse:
    query: str
    results: tuple[SearchResult, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "results": [result.as_dict() for result in self.results],
        }


class ChromeClient(Protocol):
    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ChromeToolResult: ...


def _normalize_query(query: str) -> str:
    return " ".join(query.split())


def _is_google_url(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host == "google.com" or host.endswith(".google.com")


def _pages_from_result(result: ChromeToolResult) -> list[BrowserPage]:
    structured = result.structured_content
    if isinstance(structured, dict):
        raw_pages = structured.get("pages")
        if isinstance(raw_pages, list):
            pages: list[BrowserPage] = []
            for item in raw_pages:
                if not isinstance(item, dict):
                    continue
                page_id = item.get("id")
                url = item.get("url")
                if isinstance(page_id, int) and isinstance(url, str):
                    pages.append(
                        BrowserPage(
                            id=page_id,
                            url=url,
                            selected=item.get("selected") is True,
                        )
                    )
            if pages:
                return pages

    pages = []
    in_pages_section = False
    for raw_line in result.content.splitlines():
        line = raw_line.strip()
        if line == "## Pages":
            in_pages_section = True
            continue
        if line.startswith("## ") and in_pages_section:
            break
        if not in_pages_section:
            continue
        match = _PAGE_LINE_RE.match(line)
        if not match:
            continue
        url = match.group(2).removesuffix(" [selected]").strip()
        pages.append(
            BrowserPage(
                id=int(match.group(1)),
                url=url,
                selected=line.endswith("[selected]"),
            )
        )
    return pages


def _evaluate_payload(content: str) -> dict[str, Any]:
    match = _EVALUATE_JSON_RE.search(content)
    if match is None:
        raise WebSearchError("Chrome DevTools returned an unreadable search result payload.")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise WebSearchError(
            "Chrome DevTools returned invalid JSON while extracting Google results."
        ) from exc
    if not isinstance(payload, dict):
        raise WebSearchError("Google result extraction returned an unexpected payload.")
    return payload


def _response_from_payload(
    query: str,
    payload: dict[str, Any],
    *,
    limit: int,
) -> WebSearchResponse:
    status = payload.get("status")
    if status == "blocked":
        reason = payload.get("reason")
        detail = reason if isinstance(reason, str) and reason else "Google blocked the search."
        raise WebSearchError(detail)
    if status != "ok":
        reason = payload.get("reason")
        detail = reason if isinstance(reason, str) and reason else f"status={status!r}"
        raise WebSearchError(f"Google search failed: {detail}")

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise WebSearchError("Google result extraction did not return a result list.")

    results: list[SearchResult] = []
    seen_urls: set[str] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        url = item.get("url")
        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            continue
        if url in seen_urls:
            continue

        source = item.get("source")
        if not isinstance(source, str) or not source.strip():
            source = (urlparse(url).hostname or "").removeprefix("www.")
        snippet = item.get("snippet")
        if not isinstance(snippet, str) or not snippet.strip():
            snippet = None

        seen_urls.add(url)
        results.append(
            SearchResult(
                rank=len(results) + 1,
                title=title.strip(),
                source=source.strip(),
                url=url,
                snippet=snippet.strip() if snippet is not None else None,
            )
        )
        if len(results) >= limit:
            break

    return WebSearchResponse(query=query, results=tuple(results))


class WebSearchService:
    """Google web search backed by a private, reusable headless Chrome instance."""

    def __init__(
        self,
        *,
        client_factory: Callable[[], ChromeClient] = ChromeDevToolsClient,
        result_limit: int = DEFAULT_RESULT_LIMIT,
    ) -> None:
        if result_limit <= 0:
            raise ValueError("result_limit must be positive")
        self._client_factory = client_factory
        self._result_limit = result_limit
        self._client: ChromeClient | None = None
        self._lock = asyncio.Lock()

    async def close(self) -> None:
        async with self._lock:
            await self._close_client()

    async def _close_client(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.close()

    async def _get_client(self) -> ChromeClient:
        if self._client is None:
            client = self._client_factory()
            await client.start()
            self._client = client
        return self._client

    async def _call(
        self,
        client: ChromeClient,
        name: str,
        arguments: dict[str, Any],
    ) -> ChromeToolResult:
        result = await client.call_tool(name, arguments)
        if result.is_error:
            detail = result.content.strip() or "Unknown Chrome DevTools error."
            raise ChromeDevToolsError(f"Tool '{name}' failed: {detail}")
        return result

    async def _normalize_pages(self, client: ChromeClient) -> None:
        pages = _pages_from_result(await self._call(client, "list_pages", {}))
        if not pages:
            await self._call(
                client,
                "new_page",
                {"url": GOOGLE_HOME_URL, "timeout": 15_000},
            )
            pages = _pages_from_result(await self._call(client, "list_pages", {}))
        if not pages:
            raise ChromeDevToolsError("Chrome opened without an addressable page.")

        primary = next((page for page in pages if _is_google_url(page.url)), None)
        if primary is None:
            primary = next((page for page in pages if page.selected), pages[0])

        if not primary.selected:
            await self._call(
                client,
                "select_page",
                {"pageId": primary.id, "bringToFront": False},
            )

        for page in pages:
            if page.id == primary.id:
                continue
            await self._call(client, "close_page", {"pageId": page.id})

    async def search(self, query: str) -> WebSearchResponse:
        normalized_query = _normalize_query(query)
        if not normalized_query:
            raise WebSearchError("web_search requires a non-empty query.")

        async with self._lock:
            try:
                client = await self._get_client()
                await self._normalize_pages(client)
                search_url = f"{GOOGLE_SEARCH_URL}?{urlencode({'q': normalized_query, 'num': 10})}"
                await self._call(
                    client,
                    "navigate_page",
                    {
                        "type": "url",
                        "url": search_url,
                        "timeout": 15_000,
                    },
                )
                extracted = await self._call(
                    client,
                    "evaluate_script",
                    {"function": GOOGLE_RESULTS_EXTRACTOR},
                )
                payload = _evaluate_payload(extracted.content)
                return _response_from_payload(
                    normalized_query,
                    payload,
                    limit=self._result_limit,
                )
            except ChromeDevToolsError as exc:
                await self._close_client()
                raise WebSearchError(f"Browser search failed: {exc}") from exc

"""Live news and social hooks for a forecasting question, plus the per-question state file.

A hook is {"source": "news" | "social", "query": "..."}. Polling a "news" hook asks
Google News (about 100 headlines from dozens of publishers per query; `site:x.com` in the
query narrows it to one outlet) and Bing News (fewer items, but each has a text snippet).
A "social" hook asks Bluesky and Hacker News. All four are free and need no API key.
No LLM code lives here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests

logger = logging.getLogger(__name__)

STATE_DIR = Path(os.getenv("BOT_STATE_DIR") or "state")
PER_FEED = 40  # items kept per feed per query, so one broad query cannot crowd out the rest
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; metaculus-forecast-bot)"}


def _date(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        parsed = (
            datetime.fromisoformat(text.replace("Z", "+00:00"))
            if text[:4].isdigit()
            else parsedate_to_datetime(text)
        )
    except (ValueError, TypeError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _item(title: str, url: str, source: str, published: str | None, snippet: str = "") -> dict:
    title = " ".join((title or "").split())
    when = _date(published)
    return {
        # Same headline from two feeds gets the same id, whatever the punctuation or case.
        "id": hashlib.sha1(re.sub(r"\W+", "", title.lower()).encode()).hexdigest()[:12],
        "title": title,
        "url": url or "",
        "source": source,
        "published": when.astimezone(timezone.utc).isoformat() if when else None,
        "snippet": " ".join((snippet or "").split())[:300],
    }


def _get(url: str, **params) -> requests.Response:
    response = requests.get(url, params=params, headers=_HEADERS, timeout=15)
    response.raise_for_status()
    return response


def _rss(response: requests.Response) -> list[dict]:
    # Only ever called on the fixed Google and Bing hosts below, never on a URL a model
    # chose, so the stdlib XML parser is acceptable here.
    return [
        {child.tag.rsplit("}", 1)[-1].lower(): child.text or "" for child in entry}
        for entry in ET.fromstring(response.content).iter("item")
    ]


def _google_news(query: str, days: int, until: datetime | None) -> list[dict]:
    window = f"when:{days}d"
    if until:  # a past window, for backtests
        window = f"after:{until - timedelta(days=days):%Y-%m-%d} before:{until + timedelta(days=1):%Y-%m-%d}"
    feed = _get("https://news.google.com/rss/search", q=f"{query} {window}", hl="en-US", gl="US", ceid="US:en")
    return [
        _item(
            entry.get("title", "").removesuffix(f" - {entry.get('source', '')}"),
            entry.get("link", ""),
            entry.get("source") or "Google News",
            entry.get("pubdate"),
        )
        for entry in _rss(feed)
    ]


def _bing_news(query: str, days: int, until: datetime | None) -> list[dict]:
    if until:
        return []  # Bing only serves the latest news, so it has nothing to say about a past date
    feed = _get("https://www.bing.com/news/search", q=query, format="rss", qft='sortbydate="1"')
    return [
        _item(
            entry.get("title", ""),
            parse_qs(urlparse(entry.get("link", "")).query).get("url", [entry.get("link", "")])[0],
            entry.get("source") or "Bing News",
            entry.get("pubdate"),
            entry.get("description", ""),
        )
        for entry in _rss(feed)
    ]


def _bluesky(query: str, days: int, until: datetime | None) -> list[dict]:
    stamp = "%Y-%m-%dT%H:%M:%SZ"
    window = {"since": ((until or datetime.now(timezone.utc)) - timedelta(days=days)).strftime(stamp)}
    if until:
        window["until"] = until.strftime(stamp)
    posts = _get(
        "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts", q=query, sort="top", limit=25, **window
    ).json()["posts"]
    items = []
    for post in posts:
        handle, text = post["author"]["handle"], post["record"].get("text", "")
        link = f"https://bsky.app/profile/{handle}/post/{post['uri'].rsplit('/', 1)[-1]}"
        items.append(
            _item(text[:140], link, f"@{handle} on Bluesky", post["record"].get("createdAt"), text)
        )
    return items


def _hacker_news(query: str, days: int, until: datetime | None) -> list[dict]:
    end = until or datetime.now(timezone.utc)
    window = f"created_at_i>{int((end - timedelta(days=days)).timestamp())},created_at_i<{int(end.timestamp())}"
    hits = _get(
        "https://hn.algolia.com/api/v1/search", query=query, tags="story", hitsPerPage=20, numericFilters=window
    ).json()["hits"]
    return [
        _item(
            hit.get("title") or "",
            hit.get("url") or f"https://news.ycombinator.com/item?id={hit['objectID']}",
            "Hacker News",
            hit.get("created_at"),
        )
        for hit in hits
    ]


_FEEDS = {"news": (_google_news, _bing_news), "social": (_bluesky, _hacker_news)}


def poll(hook_list: list[dict], days: int, until: datetime | None = None) -> list[dict]:
    """Fetch every hook. Returns de-duplicated items no older than `days`, newest first.

    A feed that fails is logged and skipped, so one blocked source never stops a run.
    `until` replays a past moment for backtests: only items published in the `days` before
    it are returned, and an item with no date is dropped because it cannot be placed in time.
    """
    jobs = [(feed, hook["query"]) for hook in hook_list for feed in _FEEDS.get(hook["source"], ())]

    def fetch(job) -> list[dict]:
        feed, query = job
        try:
            return feed(query, days, until)[:PER_FEED]
        except Exception as error:
            logger.warning(f"Hook {feed.__name__.strip('_')} {query!r} failed: {error}")
            return []

    end = until or datetime.now(timezone.utc)
    cutoff = (end - timedelta(days=days)).isoformat()
    found: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:  # small on purpose: stay polite to the feeds
        for batch in pool.map(fetch, jobs):
            for item in batch:
                if not item["title"] or (item["published"] or cutoff) < cutoff:
                    continue
                if until and not (item["published"] and item["published"] <= until.isoformat()):
                    continue  # published after the replayed moment, or undated
                kept = found.setdefault(item["id"], item)
                kept["snippet"] = kept["snippet"] or item["snippet"]
                if "news.google.com" in kept["url"] and item["url"]:
                    kept["url"] = item["url"]  # a publisher link can be read later; Google's wrapper cannot
    return sorted(found.values(), key=lambda item: item["published"] or "", reverse=True)


def remember(
    state: dict, scored: list[dict], now: datetime | None = None, max_age_days: int = 45, keep: int = 60
) -> None:
    """Mark scored items as seen and keep the most relevant recent ones as evidence."""
    state["seen"] = (state["seen"] + [item["id"] for item in scored])[-3000:]
    cutoff = ((now or datetime.now(timezone.utc)) - timedelta(days=max_age_days)).isoformat()
    pool = state["evidence"] + [item for item in scored if item["score"] >= 1]
    pool = [item for item in pool if (item["published"] or cutoff) >= cutoff]
    # Whole-number relevance tier first, then newest, so a fresh item outranks an older peer.
    pool.sort(key=lambda item: (int(item["score"] + 0.5), item["published"] or ""), reverse=True)
    state["evidence"] = pool[:keep]


def digest(evidence: list[dict], limit: int = 40) -> str:
    """Evidence as a prompt-ready list, most relevant first."""
    if not evidence:
        return "No relevant items found."
    lines = []
    for item in evidence[:limit]:
        title, snippet = item["title"], item["snippet"]
        # A social post's title is the start of its own text, so show the longer text once.
        text = snippet if snippet.startswith(title) else f"{title} | {snippet}" if snippet else title
        lines.append(f"- [{(item['published'] or 'undated')[:10]}] {item['source']}: {text}")
    return "\n".join(lines)


def load_state(key: int | str) -> dict:
    state = {
        "hooks": [], "hooks_built_at": None,  # the watchlist the scout wrote, and when
        "seen": [],  # ids of every item already triaged
        "evidence": [],  # scored items shown to the forecaster
        "pending": [],  # material headlines that arrived since the last forecast
        "fast_read": None,  # Jev's instant estimate, shown only to the critic
        "last_forecast_at": None, "last_prediction": None, "history": [],
    }
    path = STATE_DIR / f"{key}.json"
    if path.exists():
        try:
            state.update(json.loads(path.read_text(encoding="utf-8")))
        except ValueError as error:  # state is a cache; a corrupt file just means starting over
            logger.warning(f"Ignoring unreadable state file {path}: {error}")
    return state


def save_state(key: int | str, state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / f"{key}.json"
    scratch = path.with_suffix(".tmp")
    scratch.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
    scratch.replace(path)

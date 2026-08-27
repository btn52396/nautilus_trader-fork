"""
Minimal Massive.com (formerly Polygon.io) REST helpers for gap research.

Uses the MASSIVE_API_KEY from the repo .env. All responses are cached on disk
under data/ so re-runs don't re-hit the API.
"""

from __future__ import annotations

import gzip
import json
import threading
import time
from pathlib import Path

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / "data"
BASE_URL = "https://api.massive.com"

_thread_local = threading.local()


def load_api_key() -> str:
    for line in (REPO_ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line.startswith("MASSIVE_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("MASSIVE_API_KEY not found in .env")


API_KEY = load_api_key()


def _session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers["Authorization"] = f"Bearer {API_KEY}"
        _thread_local.session = s
    return _thread_local.session


def _get(path: str, params: dict | None = None, max_retries: int = 6) -> dict:
    url = f"{BASE_URL}{path}"
    backoff = 2.0
    for attempt in range(max_retries):
        try:
            resp = _session().get(url, params=params, timeout=60)
        except requests.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff *= 2
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(backoff)
            backoff *= 2
            continue
        raise RuntimeError(f"GET {path} -> {resp.status_code}: {resp.text[:300]}")
    raise RuntimeError(f"GET {path} failed after {max_retries} retries (rate limited?)")


def _cache_path(kind: str, name: str) -> Path:
    p = DATA_DIR / kind / f"{name}.json.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _cached(kind: str, name: str, fetch) -> dict:
    path = _cache_path(kind, name)
    if path.exists():
        with gzip.open(path, "rt") as f:
            return json.load(f)
    data = fetch()
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(data, f)
    tmp.replace(path)
    return data


def grouped_daily(date: str) -> dict:
    """
    All US stocks daily OHLCV (regular session, split-adjusted) for one date.
    """
    return _cached(
        "grouped_daily",
        date,
        lambda: _get(
            f"/v2/aggs/grouped/locale/us/market/stocks/{date}",
            {"adjusted": "true"},
        ),
    )


def grouped_daily_raw(date: str) -> dict:
    """
    All US stocks daily OHLCV (regular session, unadjusted/as-traded) for one date.

    Used to rescale adjusted aggregates back to time-of-trade prices: price
    floors, share counts, and per-share commissions must not depend on splits
    that happen after the trade date.
    """
    return _cached(
        "grouped_daily_raw",
        date,
        lambda: _get(
            f"/v2/aggs/grouped/locale/us/market/stocks/{date}",
            {"adjusted": "false"},
        ),
    )


def minute_aggs(ticker: str, date_from: str, date_to: str) -> dict:
    """
    1-minute aggregates (split-adjusted, incl. pre/post market) for a ticker.

    Timestamps in the response are bar-start Unix ms (UTC).
    """
    safe = ticker.replace("/", "_")
    return _cached(
        "minute_aggs",
        f"{safe}_{date_from}_{date_to}",
        lambda: _get(
            f"/v2/aggs/ticker/{ticker}/range/1/minute/{date_from}/{date_to}",
            {"adjusted": "true", "sort": "asc", "limit": 50000},
        ),
    )

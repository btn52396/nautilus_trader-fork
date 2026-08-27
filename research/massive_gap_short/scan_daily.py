"""
Download grouped-daily bars (all US stocks) for a date range into the local cache.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from datetime import timedelta

import massive_api as m


def weekdays(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def fetch_one(d: date) -> tuple[str, int]:
    data = m.grouped_daily(d.isoformat())
    return d.isoformat(), data.get("resultsCount", 0)


if __name__ == "__main__":
    start = date.fromisoformat(sys.argv[1])
    end = date.fromisoformat(sys.argv[2])
    days = list(weekdays(start, end))
    print(f"Fetching grouped daily for {len(days)} weekdays {start} .. {end}")
    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for iso, n in pool.map(fetch_one, days):
            done += 1
            if done % 50 == 0 or n == 0:
                print(f"  [{done}/{len(days)}] {iso}: {n} tickers")
    print("Done.")

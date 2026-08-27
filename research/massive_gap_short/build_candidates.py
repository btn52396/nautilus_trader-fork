"""
Build rough gap candidates from cached grouped-daily data.

A candidate is a (date, ticker) where, vs the previous trading day close:
  open ratio >= 1.15 OR high ratio >= 1.35
with basic garbage filters. The precise 100%+ *overnight* gap condition is
checked later against extended-hours minute data (fetch_overnight.py).

Streams consecutive day-pairs so a 10-year range stays memory-light.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"

MIN_PREV_CLOSE = 0.05
MIN_DOLLAR_VOL = 200_000.0
OPEN_RATIO = 1.15
HIGH_RATIO = 1.35


def load_day(path: Path) -> tuple[str, dict[str, dict]]:
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    date = path.name.split(".")[0]
    return date, {r["T"]: r for r in data.get("results") or []}


def main() -> None:
    files = sorted((DATA_DIR / "grouped_daily").glob("*.json.gz"))
    rows = []
    n_days = 0
    prev_date: str | None = None
    prev: dict[str, dict] = {}

    for p in files:
        date, cur = load_day(p)
        if not cur:
            continue  # holiday
        n_days += 1
        if prev_date is not None:
            for ticker, bar in cur.items():
                pbar = prev.get(ticker)
                if pbar is None:
                    continue  # IPO / first day, no prior close
                pc = pbar.get("c") or 0.0
                if pc < MIN_PREV_CLOSE:
                    continue
                o, h, c, v = bar.get("o"), bar.get("h"), bar.get("c"), bar.get("v", 0.0)
                vw = bar.get("vw") or c
                if not o or not h or not c:
                    continue
                if v * vw < MIN_DOLLAR_VOL:
                    continue
                open_ratio = o / pc
                high_ratio = h / pc
                if open_ratio >= OPEN_RATIO or high_ratio >= HIGH_RATIO:
                    rows.append(
                        {
                            "date": date,
                            "prev_date": prev_date,
                            "ticker": ticker,
                            "prev_close": pc,
                            "open": o,
                            "high": h,
                            "low": bar.get("l"),
                            "close": c,
                            "volume": v,
                            "vwap": vw,
                            "prev_volume": pbar.get("v", 0.0),
                            "open_ratio": open_ratio,
                            "high_ratio": high_ratio,
                        },
                    )
        prev_date, prev = date, cur

    df = pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)
    df.to_parquet(DATA_DIR / "candidates.parquet", index=False)
    print(f"{n_days} trading days loaded ({files[0].name.split('.')[0]} .. {files[-1].name.split('.')[0]})")
    print(f"{len(df)} rough candidates")
    print(f"  open_ratio >= 2.0 (opened 100%+ up): {(df.open_ratio >= 2.0).sum()}")
    print(f"  high_ratio >= 2.0 (touched 100%+ in RTH): {(df.high_ratio >= 2.0).sum()}")
    print(f"  per day: {len(df) / max(n_days - 1, 1):.1f}")
    print(df.groupby(df["date"].str[:4]).size().to_string())


if __name__ == "__main__":
    main()

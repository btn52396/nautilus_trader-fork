"""
Close the candidate-funnel completeness gap (lookahead audit, finding 3).

The rev-1 funnel required open >= 1.15x or day-high >= 1.35x prior close, so
events that hit 2x overnight but faded below 1.15x by the open and never
squeezed to 1.35x intraday were never fetched: the dataset keeps that zone's
losers (in-sample via the day-high arm) and is missing its winners.

This script re-scans a date range with a broadened, completeness-oriented arm:

    prev_close in [0.05, 50] and day dollar volume >= $500k and
    (volume >= 5x prior day OR open_ratio >= 1.05 OR high_ratio >= 1.15)
    and NOT already caught by the rev-1 funnel

then fetches extended-hours minute bars for the new pairs and computes the
same overnight stats as fetch_overnight.py (imported, not reimplemented).
Results land in a separate artifact; nothing existing is overwritten.

Usage: python rescan_funnel.py [since_date] [workers]
Output: data/rescan_funnel.parquet (resume-safe, saved incrementally)
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import build_candidates as bc
import fetch_overnight as fo

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT = DATA_DIR / "rescan_funnel.parquet"

MAX_PREV_CLOSE = 50.0
MIN_DOLLAR_VOL = 500_000.0
VOL_SPIKE = 5.0
SAVE_EVERY = 500


def broadened_pairs(since: str) -> list[dict]:
    files = sorted((DATA_DIR / "grouped_daily").glob("*.json.gz"))
    already = pd.read_parquet(DATA_DIR / "candidates.parquet")[["date", "ticker"]]
    already_keys = set(zip(already["date"], already["ticker"]))

    rows: list[dict] = []
    prev_date: str | None = None
    prev: dict[str, dict] = {}
    for p in files:
        date, cur = bc.load_day(p)
        if not cur:
            continue
        if prev_date is not None and date >= since:
            for ticker, bar in cur.items():
                pbar = prev.get(ticker)
                if pbar is None:
                    continue
                pc = pbar.get("c") or 0.0
                if not (bc.MIN_PREV_CLOSE <= pc <= MAX_PREV_CLOSE):
                    continue
                o, h, c, v = bar.get("o"), bar.get("h"), bar.get("c"), bar.get("v", 0.0)
                vw = bar.get("vw") or c
                if not o or not h or not c or v * vw < MIN_DOLLAR_VOL:
                    continue
                open_ratio, high_ratio = o / pc, h / pc
                if open_ratio >= bc.OPEN_RATIO or high_ratio >= bc.HIGH_RATIO:
                    continue  # rev-1 funnel already caught it
                pv = pbar.get("v", 0.0) or 0.0
                if not (v >= VOL_SPIKE * max(pv, 1.0) or open_ratio >= 1.05 or high_ratio >= 1.15):
                    continue
                if (date, ticker) in already_keys:
                    continue
                rows.append({
                    "date": date, "prev_date": prev_date, "ticker": ticker,
                    "prev_close": pc, "open": o, "high": h, "low": bar.get("l"),
                    "close": c, "volume": v, "vwap": vw, "prev_volume": pv,
                    "open_ratio": open_ratio, "high_ratio": high_ratio,
                })
        prev_date, prev = date, cur
    return rows


def main() -> None:
    since = sys.argv[1] if len(sys.argv) > 1 else "2026-05-26"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 16

    todo = broadened_pairs(since)
    print(f"broadened funnel since {since}: {len(todo)} new pairs "
          f"({len(todo) / max(len(set(r['date'] for r in todo)), 1):.0f}/day)", flush=True)

    done: set[tuple[str, str]] = set()
    rows: list[dict] = []
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        rows = prev.to_dict("records")
        done = set(zip(prev["date"], prev["ticker"]))
        print(f"resuming: {len(done)} already fetched", flush=True)
    todo = [r for r in todo if (r["date"], r["ticker"]) not in done]
    print(f"fetching {len(todo)} pairs with {workers} workers", flush=True)

    n_new = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, res in enumerate(pool.map(fo.analyze, todo), 1):
            rows.append(res)
            n_new += 1
            if i % SAVE_EVERY == 0:
                pd.DataFrame(rows).to_parquet(OUT, index=False)
                n_ev = sum(1 for r in rows if r.get("is_event"))
                print(f"  [{i}/{len(todo)}] events so far: {n_ev}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)

    ev = df[df["is_event"] == True]  # noqa: E712 - column may contain NaN
    print(f"\nRESCAN DONE: {len(df)} pairs fetched, {len(ev)} hit 2x overnight", flush=True)
    if len(ev):
        q = ev[(ev["pm_dollar_vol"] >= 1_000_000) & (ev["open"] >= 1.0)]
        print(f"qualifying (pm >= $1M, open >= $1 adj): {len(q)}")
        if len(q):
            ret = (q["open"] - q["close"]) / q["open"]
            print(f"open->close short ret: mean {ret.mean():+.1%} | median {ret.median():+.1%} | win {(ret > 0).mean():.0%}")
            print(q[["date", "ticker", "prev_close", "open", "on_high", "pm_dollar_vol"]].to_string(index=False))


if __name__ == "__main__":
    main()

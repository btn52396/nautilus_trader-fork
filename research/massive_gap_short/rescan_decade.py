"""
Decade-wide confirmation of the open >= 1.15x rule (below-the-gate side).

The 2026 pilot (rescan_funnel.py) measured the sub-1.15x-open zone over three
months only. This scan finds every remaining candidate for a 2x-overnight
event that faded below 1.15x by the open, across the full grouped-daily
history, using only NECESSARY conditions validated on the 103 known zone
events (47 pilot + 56 day-high-arm leaks):

  * day dollar volume >= $950k
      pm_dollar_vol >= $1M implies day $vol >= pm $vol (daily volume contains
      premarket: 0 violations in 21,134 pilot fetches, min ratio 1.06)
  * RAW prev close in [0.05, 500]
      the pilot capped ADJUSTED prev close at $50, which misses real zone
      events twice over: adjusted history blows past $50 via later reverse
      splits (JZXN 2023-05-01 adj $2,433 / raw ~$1.5), and genuinely
      high-priced names have zone events too (BGL 2025-07-02 raw $71.95; the
      traded universe reaches $276 - GME, 2020 leveraged ETFs). Cap applies
      in raw space via grouped_daily_raw factors.
  * volume >= 3x the ticker's LAGGED baseline (median of sessions -25..-6),
    or no baseline volume in that window (fresh listings pass)
      every $1M-premarket gapper is abnormally active vs its pre-pump self:
      the 103 known zone events measure min 5.5x / median 221x, and the lag
      makes day-2+-of-pump names (spike < 1 vs yesterday: VSA, FCHL, JLHL,
      BAOS) still spike vs the pre-pump window. A spike-vs-yesterday arm is
      NOT necessary and an is-active-yesterday arm floods the screen with the
      entire liquid small-cap tape (8.9M pairs - measured, rejected).
      Residual hole: a name elevated continuously for >20 sessions before the
      gap; judged negligible (even GME/AMC baselines were mostly pre-pump).
  * open_ratio in [0.45, 1.15), high_ratio < 1.35
      above either bound the rev-1 funnel already fetched it; the floor has
      36% margin under the weakest known zone event (0.704) - a 2x spiker
      opening below 0.45x prior close faded 78%+ from its overnight high
  * not already fetched (candidates.parquet, rescan_funnel.parquet)

Then fetches extended-hours minute bars (fetch_overnight.analyze) for the
surviving pairs. Resume-safe; appends nothing to existing artifacts.

Usage: python rescan_decade.py count                          (screen + counts)
       python rescan_decade.py fetch [workers] [sample_pct]   (fetch pairs)

sample_pct (default 100) keeps a deterministic pseudo-random subset of pairs
via crc32(date|ticker): stable across resumes, uniform within every stratum,
and a strict subset of any larger percentage - so a sample can be extended to
a census later without refetching.

Output: data/rescan_decade.parquet (+ rescan_decade_pairs.parquet screen cache)
"""

from __future__ import annotations

import gzip
import json
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import fetch_overnight as fo

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT = DATA_DIR / "rescan_decade.parquet"

MIN_RAW_PC, MAX_RAW_PC = 0.05, 500.0
MIN_DAY_DV = 950_000.0
BASE_SPIKE = 3.0          # vs lagged-median baseline volume
BASE_WINDOW, BASE_LAG = 20, 5
MIN_OPEN_RATIO = 0.45
OPEN_ARM, HIGH_ARM = 1.15, 1.35  # rev-1 funnel arms
SAVE_EVERY = 500

# Fetch-time cutoffs on the screened superset. Tightest bounds that keep 100%
# recall on the 103 known zone events (mins there: ratio 5.5x, day_dv $1.83M);
# 1.1M screened pairs -> 491k fetchable.
FETCH_RATIO = 5.0
FETCH_DAY_DV = 1_500_000.0


def load_day(path: Path) -> tuple[str, dict[str, dict]]:
    date = path.name.split(".")[0]
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    return date, {r["T"]: r for r in data.get("results") or []}


def load_raw_close(date: str) -> dict[str, float]:
    path = DATA_DIR / "grouped_daily_raw" / f"{date}.json.gz"
    if not path.exists():
        return {}
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    return {r["T"]: r["c"] for r in (data.get("results") or []) if r.get("c")}


def screen() -> pd.DataFrame:
    files = sorted((DATA_DIR / "grouped_daily").glob("*.json.gz"))
    fetched = set()
    for name in ("candidates.parquet", "rescan_funnel.parquet"):
        p = pd.read_parquet(DATA_DIR / name)[["date", "ticker"]]
        fetched |= set(zip(p["date"], p["ticker"]))

    rows: list[dict] = []
    prev_date: str | None = None
    prev: dict[str, dict] = {}
    prev_raw: dict[str, float] = {}
    vol_hist: dict[str, list[float]] = {}  # last BASE_WINDOW+BASE_LAG volumes per ticker
    for i, path in enumerate(files):
        date, cur = load_day(path)
        if not cur:
            continue
        if prev_date is not None:
            for ticker, bar in cur.items():
                pbar = prev.get(ticker)
                if pbar is None:
                    continue
                pc = pbar.get("c") or 0.0
                if pc <= 0:
                    continue
                o, h, c, v = bar.get("o"), bar.get("h"), bar.get("c"), bar.get("v", 0.0)
                vw = bar.get("vw") or c
                if not o or not h or not c or v * vw < MIN_DAY_DV:
                    continue
                if not (MIN_OPEN_RATIO <= o / pc < OPEN_ARM) or h / pc >= HIGH_ARM:
                    continue
                hist = vol_hist.get(ticker)
                base = [b for b in (hist[:-BASE_LAG] if hist else []) if b > 0]
                base_ratio = v / float(np.median(base)) if base else float("inf")
                if base_ratio < BASE_SPIKE:
                    continue
                raw_pc = prev_raw.get(ticker)
                if raw_pc is None or not (MIN_RAW_PC <= raw_pc <= MAX_RAW_PC):
                    continue
                if (date, ticker) in fetched:
                    continue
                rows.append({
                    "date": date, "prev_date": prev_date, "ticker": ticker,
                    "prev_close": pc, "prev_close_raw": raw_pc,
                    "open": o, "high": h, "low": bar.get("l"), "close": c,
                    "volume": v, "vwap": vw, "prev_volume": pbar.get("v", 0.0) or 0.0,
                    "open_ratio": o / pc, "high_ratio": h / pc,
                    "day_dv": v * vw, "base_ratio": base_ratio,
                })
        prev_date, prev = date, cur
        prev_raw = load_raw_close(date)
        for ticker, bar in cur.items():
            hist = vol_hist.setdefault(ticker, [])
            hist.append(bar.get("v", 0.0) or 0.0)
            if len(hist) > BASE_WINDOW + BASE_LAG:
                del hist[0]
        if (i + 1) % 250 == 0:
            print(f"  scanned {i + 1}/{len(files)} days, {len(rows)} pairs", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "count"
    pairs_path = DATA_DIR / "rescan_decade_pairs.parquet"
    if pairs_path.exists():
        todo = pd.read_parquet(pairs_path)
        print(f"screened pairs loaded from cache: {len(todo)}", flush=True)
    else:
        todo = screen()
        todo.to_parquet(pairs_path, index=False)
    n_days = todo["date"].nunique() if len(todo) else 0
    print(f"\ndecade screen: {len(todo)} new pairs over {n_days} event dates "
          f"({len(todo) / max(n_days, 1):.1f}/day)", flush=True)
    if len(todo):
        n_hi = int((todo["prev_close_raw"] > 50).sum())
        print(f"raw pc <= $50: {len(todo) - n_hi}   $50-500: {n_hi}")
        print("by year:")
        print(todo["date"].str[:4].value_counts().sort_index().to_string())
    if mode != "fetch":
        return

    todo = todo[(todo["base_ratio"] >= FETCH_RATIO) & (todo["day_dv"] >= FETCH_DAY_DV)]
    print(f"fetch cutoffs (ratio >= {FETCH_RATIO}x, day_dv >= ${FETCH_DAY_DV/1e6:.1f}M): "
          f"{len(todo)} pairs", flush=True)

    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    sample_pct = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
    if sample_pct < 100.0:
        keep = todo.apply(
            lambda r: zlib.crc32(f"{r['date']}|{r['ticker']}".encode()) % 10_000 < sample_pct * 100,
            axis=1,
        )
        todo = todo[keep]
        print(f"deterministic {sample_pct:.0f}% sample: {len(todo)} pairs", flush=True)

    done: set[tuple[str, str]] = set()
    rows: list[dict] = []
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        rows = prev.to_dict("records")
        done = set(zip(prev["date"], prev["ticker"]))
        print(f"resuming: {len(done)} already fetched", flush=True)
    pending = [r for r in todo.to_dict("records") if (r["date"], r["ticker"]) not in done]
    print(f"fetching {len(pending)} pairs with {workers} workers", flush=True)

    tmp = OUT.with_suffix(".tmp.parquet")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, res in enumerate(pool.map(fo.analyze, pending), 1):
            rows.append(res)
            if i % SAVE_EVERY == 0:
                pd.DataFrame(rows).to_parquet(tmp, index=False)
                tmp.replace(OUT)  # atomic: readers never see a partial file
                n_ev = sum(1 for r in rows if r.get("is_event"))
                print(f"  [{i}/{len(pending)}] events so far: {n_ev}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(tmp, index=False)
    tmp.replace(OUT)
    ev = df[df["is_event"] == True]  # noqa: E712 - column may contain NaN
    q = ev[ev["pm_dollar_vol"] >= 1_000_000]
    print(f"\nDECADE RESCAN DONE: {len(df)} pairs, {len(ev)} hit 2x overnight, "
          f"{len(q)} with pm $vol >= $1M", flush=True)


if __name__ == "__main__":
    main()

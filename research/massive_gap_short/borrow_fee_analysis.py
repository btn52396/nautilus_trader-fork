"""
Join IBorrowDesk (Interactive Brokers) borrow-fee history onto recent gap-short
events and test whether expensive-borrow gappers fade better or squeeze harder.

The free API (https://www.iborrowdesk.com/api/ticker/{SYMBOL}) serves ~1 year
of daily fee/availability snapshots, so the join covers events from ~Sep 2025
onward. Responses are cached to disk; fetching is paced to be polite.
"""

from __future__ import annotations

import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_DIR = DATA_DIR / "iborrow"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MIN_DATE = "2025-09-01"
LOOKBACK_DAYS = 10  # max staleness of the prior fee quote


def fetch_ticker(ticker: str) -> dict | None:
    cache = CACHE_DIR / f"{ticker}.json.gz"
    if cache.exists():
        with gzip.open(cache, "rt") as f:
            return json.load(f)
    try:
        resp = requests.get(
            f"https://www.iborrowdesk.com/api/ticker/{ticker}",
            timeout=20,
            headers={"User-Agent": "gap-short-research/0.1 (personal research)"},
        )
        data = resp.json() if resp.status_code == 200 else {"daily": []}
    except Exception:
        data = {"daily": []}
    with gzip.open(cache, "wt") as f:
        json.dump(data, f)
    time.sleep(0.3)
    return data


def main() -> None:
    t = pd.read_parquet(DATA_DIR / "bt_trades.parquet")
    sub = t[t["date"] >= MIN_DATE].copy()
    tickers = sorted(sub["ticker"].unique())
    print(f"{len(sub)} events, {len(tickers)} tickers to fetch")

    fee_history: dict[str, pd.DataFrame] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        for ticker, data in zip(tickers, pool.map(fetch_ticker, tickers)):
            done += 1
            if done % 50 == 0:
                print(f"  fetched {done}/{len(tickers)}")
            daily = (data or {}).get("daily") or []
            if not daily:
                continue
            df = pd.DataFrame(daily)[["date", "fee", "available"]].dropna(subset=["date"])
            fee_history[ticker] = df.sort_values("date").reset_index(drop=True)
    print(f"tickers with IB data: {len(fee_history)}/{len(tickers)}")

    rows = []
    for ev in sub.to_dict("records"):
        hist = fee_history.get(ev["ticker"])
        rec = {
            "ticker": ev["ticker"],
            "date": ev["date"],
            "ret": ev["ret_on_notional"],
            "squeeze": ev["rth_high"] / ev["avg_px_open"] - 1 if ev["avg_px_open"] else None,
            "pm_dollar_vol": ev["pm_dollar_vol"],
            "fee_prior": None,
            "avail_prior": None,
            "fee_event": None,
            "prior_quote_date": None,
        }
        if hist is not None:
            cutoff = (pd.Timestamp(ev["date"]) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
            prior = hist[(hist["date"] < ev["date"]) & (hist["date"] >= cutoff)]
            if len(prior):
                rec["fee_prior"] = prior.iloc[-1]["fee"]
                rec["avail_prior"] = prior.iloc[-1]["available"]
                rec["prior_quote_date"] = prior.iloc[-1]["date"]
            on_day = hist[hist["date"] == ev["date"]]
            if len(on_day):
                rec["fee_event"] = on_day.iloc[0]["fee"]
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_parquet(DATA_DIR / "events_with_borrow.parquet", index=False)

    n = len(df)
    has_prior = df["fee_prior"].notna()
    print(f"\nevents: {n} | with prior-day IB fee quote: {has_prior.sum()} ({has_prior.mean():.0%})")
    print("(missing = IB had no lendable inventory quoted, ticker unknown to IB, or delisted)")

    m = df[has_prior].copy()
    bins = [0, 20, 50, 100, 300, 10_000]
    labels = ["<20%", "20-50%", "50-100%", "100-300%", ">300%"]
    m["fee_bucket"] = pd.cut(m["fee_prior"], bins=bins, labels=labels)

    print("\nShort performance by PRIOR-DAY IB borrow fee (annualized):")
    g = m.groupby("fee_bucket", observed=True).agg(
        n=("ret", "size"),
        mean_ret=("ret", "mean"),
        median_ret=("ret", "median"),
        win=("ret", lambda s: (s > 0).mean()),
        worst=("ret", "min"),
        mean_squeeze=("squeeze", "mean"),
        p90_squeeze=("squeeze", lambda s: s.quantile(0.9)),
    )
    for c in ["mean_ret", "median_ret", "win", "worst", "mean_squeeze", "p90_squeeze"]:
        g[c] = (g[c] * 100).round(1)
    print(g.to_string())

    # Events IB couldn't quote at all (the censored, most-crowded group)
    miss = df[~has_prior]
    print(f"\nNo IB quote group: n={len(miss)}  mean ret {miss['ret'].mean():+.1%}  "
          f"median {miss['ret'].median():+.1%}  win {(miss['ret']>0).mean():.0%}  "
          f"worst {miss['ret'].min():+.1%}")

    corr = m[["fee_prior", "ret"]].corr().iloc[0, 1]
    print(f"\ncorr(prior fee, short return): {corr:+.3f}")
    lo = m[m["fee_prior"] <= m["fee_prior"].median()]["ret"]
    hi = m[m["fee_prior"] > m["fee_prior"].median()]["ret"]
    print(f"below-median fee: mean {lo.mean():+.1%} (n={len(lo)}) | above-median: mean {hi.mean():+.1%} (n={len(hi)})")


if __name__ == "__main__":
    main()

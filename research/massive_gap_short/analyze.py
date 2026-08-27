"""
Research statistics for the 100%+ overnight gapper short (open -> close).

short_ret is the return per $1 of short notional, before costs:
    (open - close) / open
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"


def describe(ev: pd.DataFrame, label: str) -> dict:
    r = ev["short_ret"]
    d = {
        "label": label,
        "n": int(len(ev)),
        "win_rate": float((r > 0).mean()),
        "mean": float(r.mean()),
        "median": float(r.median()),
        "std": float(r.std()),
        "p05": float(r.quantile(0.05)),
        "p95": float(r.quantile(0.95)),
        "worst": float(r.min()),
        "best": float(r.max()),
        "mean_max_squeeze": float(ev["max_squeeze"].mean()),
        "p95_max_squeeze": float(ev["max_squeeze"].quantile(0.95)),
        "worst_max_squeeze": float(ev["max_squeeze"].max()),
        "total_pnl_10k": float((r * 10_000).sum()),
    }
    return d


def bucket_table(ev: pd.DataFrame, col: str, bins: list, labels: list[str]) -> pd.DataFrame:
    ev = ev.copy()
    ev["bucket"] = pd.cut(ev[col], bins=bins, labels=labels)
    g = ev.groupby("bucket", observed=True)["short_ret"]
    out = pd.DataFrame(
        {
            "n": g.size(),
            "win_rate": g.apply(lambda x: (x > 0).mean()),
            "mean": g.mean(),
            "median": g.median(),
            "worst": g.min(),
        },
    )
    return out


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "candidates_overnight.parquet")
    ev = df[df["is_event"].fillna(False).astype(bool)].copy()

    ev["short_ret"] = (ev["open"] - ev["close"]) / ev["open"]
    ev["gap_at_open"] = ev["open"] / ev["prev_close"] - 1.0
    ev["on_gap_max"] = ev["on_high"] / ev["prev_close"] - 1.0
    # Max adverse excursion for a short from the official open (squeeze risk)
    ev["max_squeeze"] = (ev["rth_high"] - ev["open"]) / ev["open"]
    ev["fade_into_open"] = ev["on_high"] / ev["open"] - 1.0  # how far off the overnight high the open is
    ev["still_2x_at_open"] = ev["open"] >= 2.0 * ev["prev_close"]
    ev["month"] = ev["date"].str[:7]

    # Timestamps are ET strings like "2026-08-21 09:30:00-04:00"; slice HH:MM
    ev["rth_first_t"] = ev["rth_first_ts"].str[11:16]
    ev["rth_last_t"] = ev["rth_last_ts"].str[11:16]
    tradeable = ev[
        (ev["open"] >= 1.0)
        & (ev["pm_dollar_vol"] >= 1_000_000)
        & (ev["n_rth_bars"] >= 60)
        & (ev["rth_first_t"] <= "10:30")
        & (ev["rth_last_t"] >= "15:59")
    ]

    print("=" * 80)
    for d in [describe(ev, "ALL events"), describe(tradeable, "TRADEABLE subset")]:
        print(json.dumps(d, indent=2))

    print("\n--- by gap at open (open/prev_close - 1), tradeable ---")
    print(
        bucket_table(
            tradeable,
            "gap_at_open",
            bins=[-1, 0.25, 0.5, 1.0, 2.0, 100],
            labels=["<25%", "25-50%", "50-100%", "100-200%", ">200%"],
        ).to_string(float_format=lambda x: f"{x:.3f}"),
    )

    print("\n--- by open price, tradeable ---")
    print(
        bucket_table(
            tradeable,
            "open",
            bins=[0, 2, 5, 10, 25, 1e9],
            labels=["$1-2", "$2-5", "$5-10", "$10-25", ">$25"],
        ).to_string(float_format=lambda x: f"{x:.3f}"),
    )

    print("\n--- by premarket dollar volume, tradeable ---")
    print(
        bucket_table(
            tradeable,
            "pm_dollar_vol",
            bins=[0, 3e6, 10e6, 30e6, 1e12],
            labels=["$1-3M", "$3-10M", "$10-30M", ">$30M"],
        ).to_string(float_format=lambda x: f"{x:.3f}"),
    )

    print("\n--- still >= 2x at the open vs faded below, tradeable ---")
    g = tradeable.groupby("still_2x_at_open")["short_ret"]
    print(
        pd.DataFrame(
            {
                "n": g.size(),
                "win_rate": g.apply(lambda x: (x > 0).mean()),
                "mean": g.mean(),
                "median": g.median(),
                "worst": g.min(),
            },
        ).to_string(float_format=lambda x: f"{x:.3f}"),
    )

    print("\n--- monthly, tradeable ($10k/trade) ---")
    gm = tradeable.groupby("month")
    monthly = pd.DataFrame(
        {
            "n": gm.size(),
            "win_rate": gm["short_ret"].apply(lambda x: (x > 0).mean()),
            "pnl_10k": gm["short_ret"].sum() * 10_000,
        },
    )
    print(monthly.to_string(float_format=lambda x: f"{x:,.2f}"))

    print("\n--- worst 10 squeezes (tradeable), short_ret ---")
    cols = ["date", "ticker", "prev_close", "open", "close", "on_high", "rth_high", "gap_at_open", "short_ret", "max_squeeze", "pm_dollar_vol"]
    print(tradeable.nsmallest(10, "short_ret")[cols].to_string(float_format=lambda x: f"{x:,.3f}"))

    ev.to_parquet(DATA_DIR / "events_enriched.parquet", index=False)
    print(f"\nSaved events_enriched.parquet: {len(ev)} events, tradeable={len(tradeable)}")


if __name__ == "__main__":
    main()

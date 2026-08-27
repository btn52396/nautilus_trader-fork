"""
Decade-wide measurement of the sub-1.15x-open zone (confirms/refutes the
open >= 1.15x rule beyond the 3-month pilot).

Zone = (date, ticker) with a 2x overnight high, pm dollar volume >= $1M, raw
official open >= $1, and open < 1.15x prior close. Three strata with known
inclusion probability:

  A  day-high-arm events (high_ratio >= 1.35), candidates_overnight
       - complete decade census via the rev-1 funnel (the squeezers)
  B1 pilot fetch (rescan_funnel.parquet)
       - census of its window under the pilot screen
  B2 decade rescan sample (rescan_decade.parquet)
       - deterministic p% of the necessary-conditions screen (weight 100/p)

Every zone event is PIT-simulated with backtest_pit.simulate (identical entry
/exit/halt-resumption/commission conventions as the traded universe) and a
50% stop with pessimistic MINUTE fills (stop_loss_study conventions). Means
are reported unweighted and Horvitz-Thompson weighted; the decision metric is
the weighted zone mean vs the traded universe's +5.34% (50% stop).

Usage: python zone_analysis.py [sample_pct_of_B2]   (default 33)
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import backtest_pit as bp

DATA_DIR = bp.DATA_DIR
STOP = 0.50


def zone_filter(df: pd.DataFrame) -> pd.DataFrame:
    m = (
        df["is_event"].fillna(False).astype(bool)
        & (df["pm_dollar_vol"] >= 1_000_000)
        & (df["open"] / df["prev_close"] < 1.15)
    )
    return df[m].copy()


def stop_ret(trade: dict) -> float:
    """50% stop, pessimistic MINUTE fill, on the simulated trade's held bars."""
    bars = bp.minute_bars(trade["ticker"], trade["prev_date"], trade["date"], trade["date"])
    hold = bars[(bars["end_min"] > trade["entry_min"]) & (bars["end_min"] <= trade["exit_min"])]
    if not len(hold):
        return trade["ret_on_notional"]
    entry = trade["entry_px"]
    stop_px = entry * (1.0 + STOP)
    cm_high = np.maximum.accumulate(hold["h"].to_numpy(dtype=float))
    i = int(np.searchsorted(cm_high, stop_px, side="left"))
    if i >= len(cm_high):
        return trade["ret_on_notional"]
    fill = max(stop_px, float(hold["o"].to_numpy()[i]), float(hold["c"].to_numpy()[i]))
    com = 0.01 / trade["entry_raw"]  # $0.005/share each way on raw shares
    return (entry - fill) / entry - com


def describe(rets: np.ndarray, w: np.ndarray, label: str) -> None:
    mean_u = rets.mean()
    mean_w = float(np.average(rets, weights=w))
    var_w = float(np.average((rets - mean_w) ** 2, weights=w))
    n_eff = w.sum() ** 2 / (w**2).sum()
    se = np.sqrt(var_w / max(n_eff, 1))
    print(f"  {label:22s} n={len(rets):4d} (N_hat={w.sum():6.0f})  "
          f"mean {mean_u:+7.2%} unw / {mean_w:+7.2%} HT (se {se:.2%})  "
          f"median {np.median(rets):+7.2%}  win {(rets > 0).mean():5.1%}  "
          f"p05 {np.quantile(rets, 0.05):+7.2%}  worst {rets.min():+7.2%}")


def main() -> None:
    sample_pct = float(sys.argv[1]) if len(sys.argv) > 1 else 33.0
    w_b2 = 100.0 / sample_pct

    co = pd.read_parquet(DATA_DIR / "candidates_overnight.parquet")
    a = zone_filter(co[co["high"] / co["prev_close"] >= 1.35])
    a["stratum"], a["w"] = "A_dayhigh_census", 1.0

    b1 = zone_filter(pd.read_parquet(DATA_DIR / "rescan_funnel.parquet"))
    b1["stratum"], b1["w"] = "B1_pilot_census", 1.0

    try:
        b2 = zone_filter(pd.read_parquet(DATA_DIR / "rescan_decade.parquet"))
    except Exception as e:  # noqa: BLE001 - file may be absent or mid-write
        print(f"WARNING: rescan_decade.parquet unreadable ({e}); B2 stratum EMPTY")
        b2 = zone_filter(pd.DataFrame(columns=co.columns))
    b2["stratum"], b2["w"] = "B2_decade_sample", w_b2

    cols = ["date", "prev_date", "ticker", "prev_close", "open", "on_high",
            "pm_dollar_vol", "stratum", "w"]
    zone = pd.concat([a[cols], b1[cols], b2[cols]], ignore_index=True)
    zone = zone.drop_duplicates(subset=["date", "ticker"], keep="first")
    print(f"zone events by stratum:\n{zone['stratum'].value_counts().to_string()}")

    half_days = bp.detect_half_days(co)
    dates = sorted(zone["date"].unique())
    needed: dict[str, set[str]] = {}
    for d, tk in zip(zone["date"], zone["ticker"]):
        needed.setdefault(d, set()).add(tk)
    factors, n_missing = bp.build_factors(dates, needed)
    print(f"factors: {len(factors)} (missing {n_missing})")
    all_dates = bp.trading_dates()

    rows = zone.to_dict("records")
    results = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = [pool.submit(bp.simulate, r, half_days, factors, all_dates) for r in rows]
        for r, f in zip(rows, futs):
            t = f.result()
            t["stratum"], t["w"] = r["stratum"], r["w"]
            t["prev_date"] = r["prev_date"]
            results.append(t)

    traded = [
        (r, t) for r, t in zip(rows, results)
        if t["traded"] and t["raw_open"] >= bp.MIN_RAW_OPEN
    ]
    print(f"\nsimulatable zone trades (raw open >= $1): {len(traded)} of {len(rows)}")

    with ThreadPoolExecutor(max_workers=16) as pool:
        stop_rets = list(pool.map(lambda rt: stop_ret(rt[1]), traded))

    df = pd.DataFrame(
        {
            "date": [t["date"] for _, t in traded],
            "ticker": [t["ticker"] for _, t in traded],
            "stratum": [t["stratum"] for _, t in traded],
            "w": [t["w"] for _, t in traded],
            "ret": [t["ret_on_notional"] for _, t in traded],
            "ret_stop": stop_rets,
            "exit_kind": [t["exit_kind"] for _, t in traded],
        },
    )
    df.to_parquet(DATA_DIR / "zone_analysis.parquet", index=False)

    print("\n--- zone edge (short, net of commissions), PIT conventions ---")
    for col, lbl in [("ret", "no stop"), ("ret_stop", "50% stop")]:
        r, w = df[col].to_numpy(), df["w"].to_numpy()
        describe(r, w, f"all zone, {lbl}")
        for s in sorted(df["stratum"].unique()):
            d = df[df["stratum"] == s]
            describe(d[col].to_numpy(), d["w"].to_numpy(), f"  {s}, {lbl}")

    print("\nby year (HT-weighted, 50% stop):")
    df["year"] = df["date"].str[:4]
    for y, g in df.groupby("year"):
        r, w = g["ret_stop"].to_numpy(), g["w"].to_numpy()
        print(f"  {y}: n={len(g):3d} N_hat={w.sum():5.0f}  mean {np.average(r, weights=w):+7.2%}  win {(r > 0).mean():5.1%}")

    print("\ntraded-universe reference (top-5, 50% stop): mean +5.34%, win 68.1%")


if __name__ == "__main__":
    main()

"""
Robustness check behind the stop-level recommendation.

The headline stop grid (stop_loss_study.py) ranks levels by compounded growth
at f = min(half Kelly, f holding the worst intraday dip to -50% of equity),
all fit on the full decade. That ranking can be flattered by in-sample luck,
so re-rank the candidate levels under:

  A. the headline convention (baseline, for reference)
  B. conservative sizing: f = min(quarter Kelly, dip floor -25% of equity)
  C. split-half pseudo-out-of-sample: fit f on one half of the decade
     (2016-2021 / 2021-2026), compound the OTHER half at that f, combine

A level is only recommendable if it ranks well under all three.

Usage: python stop_loss_robustness.py
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import backtest_pit as bp
import stop_loss_study as s


LEVELS: list[float | None] = [None, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00, 2.00]


def fit_f(units: np.ndarray, troughs: np.ndarray, kelly_div: float, dip: float) -> float:
    f_kelly = bp.kelly_opt(units)
    f_safe = dip / abs(troughs.min())
    return f_safe if np.isnan(f_kelly) else min(f_kelly / kelly_div, f_safe)


def growth_yr(units: np.ndarray, f: float, years: float) -> float:
    return float(np.expm1(np.log1p(np.clip(f * units, -0.999, None)).sum() / years))


def main() -> None:
    t = pd.read_parquet(s.DATA_DIR / "bt_trades_pit.parquet")
    cands = pd.read_parquet(s.DATA_DIR / "candidates_overnight.parquet")[["date", "ticker", "prev_date"]]
    t = t.merge(cands, on=["date", "ticker"], how="left", validate="one_to_one")
    trades = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        trades = [r for r in pool.map(s.load_trade, t.to_dict("records")) if r is not None]
    print(f"loaded {len(trades)} trades")

    rows = []
    for x in LEVELS:
        trig_key = f"trig_{x}" if x is not None else None
        ret_key = f"ret_minute_{x}" if x is not None else None
        daily = s.daily_trough(trades, ret_key, trig_key)
        dates = pd.to_datetime(daily["date"])
        units = daily["pnl"].to_numpy() / s.UNIT
        troughs = daily["trough"].to_numpy() / s.UNIT
        years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25

        f_a = fit_f(units, troughs, 2, 0.50)
        f_b = fit_f(units, troughs, 4, 0.25)

        mid = dates.iloc[0] + (dates.iloc[-1] - dates.iloc[0]) / 2
        h1 = (dates < mid).to_numpy()
        f_h1 = fit_f(units[h1], troughs[h1], 2, 0.50)   # fit on first half
        f_h2 = fit_f(units[~h1], troughs[~h1], 2, 0.50)  # fit on second half
        # compound each half at the f fitted on the other half
        log_oos = (
            np.log1p(np.clip(f_h2 * units[h1], -0.999, None)).sum()
            + np.log1p(np.clip(f_h1 * units[~h1], -0.999, None)).sum()
        )
        rows.append({
            "stop": "none" if x is None else f"{int(x*100)}%",
            "A_f": f_a, "A_growth": growth_yr(units, f_a, years),
            "B_f": f_b, "B_growth": growth_yr(units, f_b, years),
            "C_f_fit_h1": f_h1, "C_f_fit_h2": f_h2,
            "C_growth_oos": float(np.expm1(log_oos / years)),
        })

    df = pd.DataFrame(rows)
    out = df.copy()
    for c in out.columns:
        if c.endswith("_growth") or c == "C_growth_oos":
            out[c] = out[c].map("{:+.1%}".format)
        elif c != "stop":
            out[c] = out[c].map("{:.4f}".format)
    pd.set_option("display.width", 200)
    print("\nA = min(half Kelly, -50% dip) full-sample | B = min(quarter Kelly, -25% dip) | C = split-half, f fit on the other half:")
    print(out.to_string(index=False))
    df.to_parquet(s.DATA_DIR / "stop_robustness.parquet", index=False)


if __name__ == "__main__":
    main()

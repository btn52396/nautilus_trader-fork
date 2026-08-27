"""
Is open >= 1.15x prior close the right qualification threshold?

The 1.15 gate was inherited from the rev-1 candidate funnel arm, not derived
from the edge. This sweep re-evaluates the threshold under the FINAL live
configuration (top-5 names/day by premarket dollar volume, 50% stop with
pessimistic MINUTE fills, flat $10k/trade, entry-date attribution):

  * thresholds >= 1.15 use a complete universe (every open>=1.15x name was
    fetched by the rev-1 funnel in all ten years - set inclusion);
  * rows below 1.15 are labelled BIASED: only the day-high >= 1.35x arm's
    names exist below the gate, a loser-enriched sample (audit finding 3);
  * qualification is applied BEFORE the top-5 selection, so busy days refill
    from names ranked 6+ by premarket dollar volume, exactly as live would;
  * per-threshold sizing is re-derived (min of half Kelly and the f that
    holds the worst intraday trough to -50% of equity) and growth is also
    reported at the frozen production f = 0.137 for apples-to-apples;
  * split-half (2016-08..2021-08 vs 2021-08..2026-08) guards against
    picking a threshold on in-sample luck.

Usage: python open_threshold_study.py
Outputs: data/open_threshold_study.parquet + printed tables.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_pit as bp
import stop_loss_study as sls


DATA_DIR = Path(__file__).resolve().parent / "data"
UNIT = 10_000.0
STOP = 0.50
TOP_N = 5
F_PROD = 0.137
EQUITY_DIP_FLOOR = 0.50
SPLIT_DATE = "2021-08-26"
THRESHOLDS = [None, 1.10, 1.15, 1.20, 1.25, 1.35, 1.50, 1.75, 2.00]

TRIG, RET = f"trig_{STOP}", f"ret_minute_{STOP}"


def stop_ret(t: dict) -> float:
    return t[RET] if t[TRIG] >= 0 else t["ret_orig"]


def select(trades: list[dict], thr: float | None) -> list[dict]:
    """Qualify on the open ratio, then keep the top-5 by premarket $vol per day."""
    qual = [t for t in trades if thr is None or t["open_ratio"] >= thr]
    meta = pd.DataFrame(
        {"i": range(len(qual)),
         "date": [t["date"] for t in qual],
         "pm": [t["pm_dollar_vol"] for t in qual]},
    )
    keep = meta.sort_values("pm", ascending=False).groupby("date").head(TOP_N)["i"]
    return [qual[i] for i in sorted(keep)]


def growth_yr(units: np.ndarray, f: float, years: float) -> float:
    return float(np.expm1(np.log1p(np.clip(f * units, -0.999, None)).sum() / years))


def evaluate(sel: list[dict], label: str, years: float, full_range: tuple[str, str]) -> dict:
    rets = np.array([stop_ret(t) for t in sel])
    daily = sls.daily_trough(sel, RET, TRIG)
    units = daily["pnl"].to_numpy() / UNIT
    troughs = daily["trough"].to_numpy() / UNIT

    cal = pd.Series(0.0, index=pd.bdate_range(full_range[0], full_range[1]))
    cal[pd.to_datetime(daily["date"])] = units

    f_kelly = bp.kelly_opt(units)
    f_safe = EQUITY_DIP_FLOOR / abs(troughs.min())
    f_use = min(f_kelly / 2, f_safe)

    h1 = np.array([stop_ret(t) for t in sel if t["date"] < SPLIT_DATE])
    h2 = np.array([stop_ret(t) for t in sel if t["date"] >= SPLIT_DATE])

    return {
        "thr": label,
        "n": len(sel),
        "per_yr": len(sel) / years,
        "mean": float(rets.mean()),
        "median": float(np.median(rets)),
        "win": float((rets > 0).mean()),
        "p05": float(np.quantile(rets, 0.05)),
        "worst": float(rets.min()),
        "sharpe_active": float(units.mean() / units.std(ddof=1) * np.sqrt(252)),
        "sharpe_cal": float(cal.mean() / cal.std(ddof=1) * np.sqrt(252)),
        "kelly_f_day": f_kelly,
        "worst_trough": float(troughs.min()),
        "f_use": f_use,
        "growth_at_f_use": growth_yr(units, f_use, years),
        "growth_at_f137": growth_yr(units, F_PROD, years),
        "mean_h1": float(h1.mean()),
        "mean_h2": float(h2.mean()),
    }


def main() -> None:
    t = pd.read_parquet(DATA_DIR / "bt_trades_pit.parquet")
    cands = pd.read_parquet(DATA_DIR / "candidates_overnight.parquet")[["date", "ticker", "prev_date"]]
    t = t.merge(cands, on=["date", "ticker"], how="left", validate="one_to_one")
    t["open_ratio"] = t["open"] / t["prev_close"]
    print(f"PIT trades: {len(t)}")

    rows = t.to_dict("records")
    trades = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i, (row, r) in enumerate(zip(rows, pool.map(sls.load_trade, rows)), 1):
            if r is not None:
                r["open_ratio"] = row["open_ratio"]
                r["pm_dollar_vol"] = row["pm_dollar_vol"]
                trades.append(r)
            if i % 500 == 0:
                print(f"  [{i}/{len(rows)}]", flush=True)
    print(f"loaded: {len(trades)}")

    d0, d1 = t["date"].min(), t["date"].max()
    years = (pd.to_datetime(d1) - pd.to_datetime(d0)).days / 365.25

    results = []
    for thr in THRESHOLDS:
        sel = select(trades, thr)
        label = "none*" if thr is None else (f"{thr:.2f}*" if thr < 1.15 else f"{thr:.2f}")
        results.append(evaluate(sel, label, years, (d0, d1)))

    df = pd.DataFrame(results)
    df.to_parquet(DATA_DIR / "open_threshold_study.parquet", index=False)

    pd.set_option("display.width", 250)
    fmt = {
        "per_yr": "{:.0f}", "mean": "{:+.2%}", "median": "{:+.2%}", "win": "{:.1%}",
        "p05": "{:+.2%}", "worst": "{:+.2%}", "sharpe_active": "{:.2f}", "sharpe_cal": "{:.2f}",
        "kelly_f_day": "{:.3f}", "worst_trough": "{:+.2f}", "f_use": "{:.3f}",
        "growth_at_f_use": "{:+.1%}", "growth_at_f137": "{:+.1%}",
        "mean_h1": "{:+.2%}", "mean_h2": "{:+.2%}",
    }
    out = df.copy()
    for c, f in fmt.items():
        out[c] = out[c].map(lambda v, f=f: f.format(v) if pd.notna(v) else "-")
    print(f"\nTop-{TOP_N} by premarket $vol, {STOP:.0%} stop (MINUTE fills); * = biased sample below 1.15 (funnel gap):")
    print(out.to_string(index=False))

    # Marginal view: bucket the AS-TRADED final-config (1.15) set by open ratio.
    sel = select(trades, 1.15)
    bands = [(1.15, 1.25), (1.25, 1.50), (1.50, 2.00), (2.00, 3.00), (3.00, np.inf)]
    print(f"\nMarginal bands within the 1.15 config (as traded, {STOP:.0%} stop):")
    tot = sum(stop_ret(x) for x in sel)
    for lo, hi in bands:
        b = [x for x in sel if lo <= x["open_ratio"] < hi]
        r = np.array([stop_ret(x) for x in b])
        print(f"  [{lo:.2f},{hi if np.isfinite(hi) else 99:.2f}): n={len(b):5d}"
              f"  mean {r.mean():+7.2%}  median {np.median(r):+7.2%}  win {(r > 0).mean():5.1%}"
              f"  share_of_pnl {r.sum() / tot:6.1%}")


if __name__ == "__main__":
    main()

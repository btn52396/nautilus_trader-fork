"""
Compare position-sizing rules on the engine-verified per-trade returns.

Sizing rules re-weight the same trades (returns are size-independent in the
backtest, which is fair up to the participation cap each rule enforces):

  flat   : $10k per event
  linear : 0.5% of premarket dollar volume, capped at $50k per name
  sqrt   : sqrt-scaled participation (sub-linear in volume), capped at $50k

For comparability, every rule is scaled so its *average daily gross deployment*
matches the flat rule (~$37k/day). The unscaled "raw" linear rule is also
reported as a capacity estimate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"

PARTICIPATION = 0.005  # 0.5% of premarket dollar volume
CAP = 50_000.0
MIN_TICKET = 1_000.0  # skip trades the rule can't fund meaningfully


def evaluate(trades: pd.DataFrame, size: pd.Series, label: str) -> dict:
    t = trades.copy()
    t["size"] = size
    t = t[t["size"] >= MIN_TICKET]
    t["pnl_rule"] = t["size"] * t["ret_on_notional"]

    daily = t.groupby("date").agg(pnl=("pnl_rule", "sum"), gross=("size", "sum"))
    cum = daily["pnl"].cumsum()
    dd = cum - cum.cummax()
    top10 = t["size"].nlargest(int(len(t) * 0.1)).sum() / t["size"].sum()

    return {
        "rule": label,
        "n_trades": int(len(t)),
        "avg_daily_gross": float(daily["gross"].mean()),
        "total_pnl": float(t["pnl_rule"].sum()),
        "ret_per_dollar": float(t["pnl_rule"].sum() / t["size"].sum()),
        "sharpe_daily_ann": float(daily["pnl"].mean() / daily["pnl"].std() * np.sqrt(252)),
        "max_dd": float(dd.min()),
        "worst_trade": float(t["pnl_rule"].min()),
        "worst_day": float(daily["pnl"].min()),
        "top10pct_gross_share": float(top10),
    }


def main() -> None:
    trades = pd.read_parquet(DATA_DIR / "bt_trades.parquet")

    flat = pd.Series(10_000.0, index=trades.index)

    linear_raw = (trades["pm_dollar_vol"] * PARTICIPATION).clip(upper=CAP)
    sqrt_raw = np.sqrt(trades["pm_dollar_vol"].clip(lower=0.0))

    target = evaluate(trades, flat, "flat")["avg_daily_gross"]

    def normalize(raw: pd.Series) -> pd.Series:
        # Scale so average daily gross matches the flat rule, respecting the cap
        # (iterate because the cap re-binds as the scale changes).
        k = 1.0
        for _ in range(20):
            scaled = (raw * k).clip(upper=CAP)
            mask = scaled >= MIN_TICKET
            daily = scaled[mask].groupby(trades.loc[mask, "date"]).sum()
            avg = daily.reindex(trades["date"].unique()).fillna(0.0).mean()
            if abs(avg - target) / target < 0.001:
                break
            k *= target / avg
        return (raw * k).clip(upper=CAP)

    results = [
        evaluate(trades, flat, "flat $10k"),
        evaluate(trades, normalize(linear_raw), "linear 0.5% pm$vol (matched gross)"),
        evaluate(trades, normalize(sqrt_raw), "sqrt pm$vol (matched gross)"),
        evaluate(trades, linear_raw, "linear 0.5% pm$vol (raw capacity)"),
    ]

    df = pd.DataFrame(results)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    # Where do the tail losses sit relative to premarket volume?
    t = trades.copy()
    t["pm_vol_decile"] = pd.qcut(t["pm_dollar_vol"], 10, labels=False) + 1
    tail = t[t["ret_on_notional"] <= -1.0]
    print("\nTrades losing >100% of notional, by premarket $vol decile (10 = most liquid):")
    print(tail["pm_vol_decile"].value_counts().sort_index().to_string())
    print(f"\ncorr(ret, log pm$vol) = {t['ret_on_notional'].corr(np.log(t['pm_dollar_vol'])):.4f}")

    (DATA_DIR / "sizing_comparison.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

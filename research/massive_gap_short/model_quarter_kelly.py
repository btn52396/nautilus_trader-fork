"""
Compounded replay: quarter-Kelly base size with only the participation cap.

Rule per trade:  size = min( f * equity_at_day_start,  0.5% * premarket_$vol )
with f = quarter Kelly (0.0529), starting equity $40k, no per-name hard cap,
no separate risk budget. Uses the engine-verified per-trade returns; assumes
zero market impact up to the 0.5% participation cap (that is what the cap is
for) and zero borrow costs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"

# Fraction of equity per trade; default quarter Kelly (per-day Kelly 0.2118 / 4)
F = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0529
RESET_YEARLY = "--reset-yearly" in sys.argv
PARTICIPATION = 0.005
START_EQUITY = 40_000.0


def simulate(trades: pd.DataFrame, use_cap: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = []
    equity = START_EQUITY
    year = None
    for date, day in trades.groupby("date"):
        if RESET_YEARLY and date[:4] != year:
            year = date[:4]
            equity = START_EQUITY
        base = F * equity
        cap = PARTICIPATION * day["pm_dollar_vol"].to_numpy()
        size = np.minimum(base, cap) if use_cap else np.full(len(day), base)
        pnl = float((size * day["ret_on_notional"].to_numpy()).sum())
        days.append(
            {
                "date": date,
                "equity_start": equity,
                "base": base,
                "n": len(day),
                "n_capped": int((cap < base).sum()) if use_cap else 0,
                "gross": float(size.sum()),
                "pnl": pnl,
            },
        )
        equity += pnl
        if equity <= 0:
            break
    df = pd.DataFrame(days)
    df["equity"] = df["equity_start"] + df["pnl"]
    df["year"] = df["date"].str[:4]
    if RESET_YEARLY:
        peak = df.groupby("year")["equity"].cummax()
    else:
        peak = df["equity"].cummax()
    df["dd"] = df["equity"] / peak - 1.0
    yearly = df.groupby("year").agg(
        end_equity=("equity", "last"),
        pnl=("pnl", "sum"),
        n_trades=("n", "sum"),
        n_capped=("n_capped", "sum"),
        max_dd=("dd", "min"),
        avg_daily_gross=("gross", "mean"),
    )
    yearly["ret"] = yearly["pnl"] / (yearly["end_equity"] - yearly["pnl"])
    yearly["pct_capped"] = yearly["n_capped"] / yearly["n_trades"]
    return df, yearly


def main() -> None:
    trades = pd.read_parquet(DATA_DIR / "bt_trades.parquet")

    capped, yearly = simulate(trades, use_cap=True)
    pure, yearly_pure = simulate(trades, use_cap=False)

    print(f"Quarter Kelly f={F}, start ${START_EQUITY:,.0f}, 0.5% participation cap")
    print("=" * 100)
    cols = ["end_equity", "ret", "max_dd", "pct_capped", "avg_daily_gross", "n_trades"]
    out = yearly[cols].copy()
    out["end_equity"] = out["end_equity"].map(lambda x: f"${x:,.0f}")
    out["ret"] = out["ret"].map(lambda x: f"{x:+.1%}")
    out["max_dd"] = out["max_dd"].map(lambda x: f"{x:.1%}")
    out["pct_capped"] = out["pct_capped"].map(lambda x: f"{x:.0%}")
    out["avg_daily_gross"] = out["avg_daily_gross"].map(lambda x: f"${x:,.0f}")
    print(out.to_string())

    print(f"\nWith cap:    terminal ${capped['equity'].iloc[-1]:,.0f}   max DD {capped['dd'].min():.1%}")
    print(f"Without cap: terminal ${pure['equity'].iloc[-1]:,.0f}   max DD {pure['dd'].min():.1%}")

    w = capped.nsmallest(3, "pnl")[["date", "equity_start", "gross", "pnl"]]
    w["pnl_pct"] = w["pnl"] / w["equity_start"]
    print("\nWorst 3 days (capped run):")
    print(w.to_string(index=False, float_format=lambda x: f"{x:,.0f}" if abs(x) > 10 else f"{x:.1%}"))

    # Milestones for when the cap starts to matter
    for thresh in [0.25, 0.5, 0.75]:
        m = capped[capped["n_capped"] / capped["n"] >= thresh]
        if len(m):
            r = m.iloc[0]
            print(f"cap binds on >={thresh:.0%} of a day's trades first on {r['date']} (equity ${r['equity_start']:,.0f})")

    capped[["date", "equity", "dd", "base", "gross", "n", "n_capped"]].to_parquet(
        DATA_DIR / "qk_replay.parquet", index=False,
    )


if __name__ == "__main__":
    main()

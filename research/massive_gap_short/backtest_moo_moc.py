"""
MOO/MOC variant of the gap-short backtest.

Fills at the official auction prints (daily open / daily close from grouped
dailies) instead of the engine's 09:31 / 15:58 last-trade fills. Same events,
same qty = int($10k / official open), same $0.005/share commissions. Decomposes
entry vs exit effects across all four fill combinations and replays the
quarter-Kelly yearly-reset account on the MOO/MOC returns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
COMMISSION_PER_SHARE = 0.005

F = 0.0529  # quarter Kelly per trade
PARTICIPATION = 0.005
START_EQUITY = 40_000.0


def combo_returns(t: pd.DataFrame, entry_col: str, exit_col: str) -> pd.Series:
    entry = t[entry_col]
    exit_ = t[exit_col]
    pnl = t["qty"] * (entry - exit_) - 2 * t["qty"] * COMMISSION_PER_SHARE
    return pnl / (entry * t["qty"])


def summarize(t: pd.DataFrame, ret: pd.Series, label: str) -> dict:
    pnl = ret * 10_000.0
    daily = pnl.groupby(t["date"]).sum()
    cum = daily.cumsum()
    dd = (cum - cum.cummax()).min()
    return {
        "variant": label,
        "mean": ret.mean(),
        "median": ret.median(),
        "win": (ret > 0).mean(),
        "std": ret.std(),
        "worst": ret.min(),
        "total_pnl_10k": pnl.sum(),
        "sharpe": daily.mean() / daily.std() * np.sqrt(252),
        "max_dd_10k": dd,
    }


def reset_replay(t: pd.DataFrame, ret: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"date": t["date"], "ret": ret, "pm": t["pm_dollar_vol"]})
    days = []
    equity = START_EQUITY
    year = None
    for date, day in df.groupby("date"):
        if date[:4] != year:
            year = date[:4]
            equity = START_EQUITY
        size = np.minimum(F * equity, PARTICIPATION * day["pm"].to_numpy())
        pnl = float((size * day["ret"].to_numpy()).sum())
        days.append({"date": date, "equity_start": equity, "pnl": pnl})
        equity += pnl
    d = pd.DataFrame(days)
    d["equity"] = d["equity_start"] + d["pnl"]
    d["year"] = d["date"].str[:4]
    peak = d.groupby("year")["equity"].cummax()
    d["dd"] = d["equity"] / peak - 1.0
    return d.groupby("year").agg(end_equity=("equity", "last"), max_dd=("dd", "min"))


def main() -> None:
    t = pd.read_parquet(DATA_DIR / "bt_trades.parquet")

    combos = [
        ("Engine: 09:31 -> 15:58", "avg_px_open", "avg_px_close"),
        ("MOO -> 15:58", "open", "avg_px_close"),
        ("09:31 -> MOC", "avg_px_open", "close"),
        ("MOO -> MOC", "open", "close"),
    ]
    rows = []
    rets = {}
    for label, e, x in combos:
        ret = combo_returns(t, e, x)
        rets[label] = ret
        rows.append(summarize(t, ret, label))

    s = pd.DataFrame(rows).set_index("variant")
    for c in ["mean", "median", "win", "std", "worst"]:
        s[c] = (s[c] * 100).round(2)
    s["total_pnl_10k"] = s["total_pnl_10k"].round(0)
    s["max_dd_10k"] = s["max_dd_10k"].round(0)
    s["sharpe"] = s["sharpe"].round(2)
    print("Per-trade stats (%, flat $10k/trade for PnL columns):")
    print(s.to_string())

    # Exit-leg effect isolated: 15:58 -> close move captured (short covers later)
    exit_gain = rets["MOO -> MOC"] - rets["MOO -> 15:58"]
    print(f"\nExit leg (MOC vs 15:58): mean {exit_gain.mean():+.3%}  median {exit_gain.median():+.3%}  "
          f"helps {(exit_gain > 0).mean():.1%} of trades")
    entry_gain = rets["MOO -> MOC"] - rets["09:31 -> MOC"]
    print(f"Entry leg (MOO vs 09:31): mean {entry_gain.mean():+.3%}  median {entry_gain.median():+.3%}")

    print("\nQuarter-Kelly yearly-reset replay ($40k each Jan, 0.5% cap):")
    eng = reset_replay(t, rets["Engine: 09:31 -> 15:58"])
    moo = reset_replay(t, rets["MOO -> MOC"])
    cmp_ = eng.join(moo, lsuffix="_engine", rsuffix="_moomoc")
    cmp_["end_equity_engine"] = cmp_["end_equity_engine"].map(lambda x: f"${x:,.0f}")
    cmp_["end_equity_moomoc"] = cmp_["end_equity_moomoc"].map(lambda x: f"${x:,.0f}")
    cmp_["max_dd_engine"] = cmp_["max_dd_engine"].map(lambda x: f"{x:.1%}")
    cmp_["max_dd_moomoc"] = cmp_["max_dd_moomoc"].map(lambda x: f"{x:.1%}")
    print(cmp_.to_string())

    out = t[["date", "ticker", "qty", "open", "close", "pm_dollar_vol"]].copy()
    out["ret_moo_moc"] = rets["MOO -> MOC"]
    out.to_parquet(DATA_DIR / "bt_trades_moo_moc.parquet", index=False)


if __name__ == "__main__":
    main()

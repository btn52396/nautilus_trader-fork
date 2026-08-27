"""
Join NautilusTrader backtest positions to events and compute performance.

Saves bt_trades.parquet (one row per trade, event metadata + realized PnL) and
bt_summary.json (headline stats + daily equity curve) for presentation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
NOTIONAL = 10_000.0
COMMISSION_PER_SHARE = 0.005


def main() -> None:
    pos = pd.read_csv(DATA_DIR / "bt_positions.csv")
    events = pd.read_parquet(DATA_DIR / "bt_events.parquet")

    pos["ticker_sane"] = pos["instrument_id"].str.replace(".XNAS", "", regex=False)
    pos["date"] = (
        pd.to_datetime(pos["ts_opened"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%Y-%m-%d")
    )
    pos["entry_et"] = (
        pd.to_datetime(pos["ts_opened"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%H:%M")
    )
    pos["exit_et"] = (
        pd.to_datetime(pos["ts_closed"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%H:%M")
    )
    pos["gross_pnl"] = pos["realized_pnl"].str.replace(" USD", "", regex=False).astype(float)

    events = events.copy()
    events["ticker_sane"] = events["ticker"].str.replace(".", "-", regex=False).str.replace("/", "-", regex=False)

    trades = pos.merge(
        events,
        on=["ticker_sane", "date"],
        how="left",
        suffixes=("", "_ev"),
        validate="one_to_one",
    )
    assert trades["ticker"].notna().all(), "unmatched positions"

    # $0.005/share each way (engine can't represent sub-cent USD fees)
    trades["commission"] = 2 * trades["qty"] * COMMISSION_PER_SHARE
    trades["pnl"] = trades["gross_pnl"] - trades["commission"]
    trades["entry_notional"] = trades["avg_px_open"] * trades["qty"]
    trades["ret_on_notional"] = trades["pnl"] / trades["entry_notional"]
    # Research reference: official open -> official close, no costs
    trades["ret_research"] = (trades["open"] - trades["close"]) / trades["open"]

    r = trades["ret_on_notional"]
    daily = trades.groupby("date").agg(pnl=("pnl", "sum"), n=("pnl", "size"), notional=("entry_notional", "sum"))
    daily["cum_pnl"] = daily["pnl"].cumsum()
    peak = daily["cum_pnl"].cummax()
    dd = daily["cum_pnl"] - peak

    monthly = trades.copy()
    monthly["month"] = monthly["date"].str[:7]
    mg = monthly.groupby("month").agg(pnl=("pnl", "sum"), n=("pnl", "size"))

    summary = {
        "n_trades": int(len(trades)),
        "n_days": int(len(daily)),
        "n_tickers": int(trades["ticker"].nunique()),
        "date_range": [trades["date"].min(), trades["date"].max()],
        "total_pnl": float(trades["pnl"].sum()),
        "total_commission": float(trades["commission"].sum()),
        "win_rate": float((trades["pnl"] > 0).mean()),
        "mean_ret": float(r.mean()),
        "median_ret": float(r.median()),
        "std_ret": float(r.std()),
        "p05_ret": float(r.quantile(0.05)),
        "p95_ret": float(r.quantile(0.95)),
        "worst_ret": float(r.min()),
        "worst_trade_pnl": float(trades["pnl"].min()),
        "best_trade_pnl": float(trades["pnl"].max()),
        "mean_ret_research": float(trades["ret_research"].mean()),
        "slippage_per_trade": float((trades["ret_research"] - r).mean()),
        "avg_trades_per_day": float(daily["n"].mean()),
        "max_trades_per_day": int(daily["n"].max()),
        "avg_daily_notional": float(daily["notional"].mean()),
        "max_daily_notional": float(daily["notional"].max()),
        "max_drawdown": float(dd.min()),
        "pct_days_positive": float((daily["pnl"] > 0).mean()),
        "sharpe_daily_ann": float(daily["pnl"].mean() / daily["pnl"].std() * np.sqrt(252)),
        "monthly_pnl": {k: float(v) for k, v in mg["pnl"].items()},
        "n_months_positive": int((mg["pnl"] > 0).sum()),
        "n_months": int(len(mg)),
    }

    trades_out = trades[
        [
            "date", "ticker", "qty", "avg_px_open", "avg_px_close", "entry_et", "exit_et",
            "pnl", "commission", "ret_on_notional", "ret_research", "entry_notional",
            "prev_close", "open", "close", "on_high", "rth_high", "on_gap_max",
            "pm_dollar_vol", "on_high_time",
        ]
    ].copy()
    trades_out.to_parquet(DATA_DIR / "bt_trades.parquet", index=False)
    daily.reset_index().to_parquet(DATA_DIR / "bt_daily.parquet", index=False)
    (DATA_DIR / "bt_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print("\nWorst 5 trades:")
    print(trades.nsmallest(5, "pnl")[["date", "ticker", "avg_px_open", "avg_px_close", "qty", "pnl", "ret_on_notional"]].to_string())


if __name__ == "__main__":
    main()

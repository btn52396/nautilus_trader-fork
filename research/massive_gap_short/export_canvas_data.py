"""
Export a compact JSON bundle of research + backtest results for the canvas.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"


def bucket_stats(df: pd.DataFrame, col: str, bins: list, labels: list[str]) -> list[dict]:
    out = []
    b = pd.cut(df[col], bins=bins, labels=labels)
    for label in labels:
        sub = df[b == label]
        if not len(sub):
            continue
        out.append(
            {
                "bucket": label,
                "n": int(len(sub)),
                "win_rate": round(float((sub["ret_on_notional"] > 0).mean()), 4),
                "mean": round(float(sub["ret_on_notional"].mean()), 4),
                "median": round(float(sub["ret_on_notional"].median()), 4),
                "worst": round(float(sub["ret_on_notional"].min()), 4),
                "pnl": round(float(sub["pnl"].sum()), 0),
            },
        )
    return out


def funnel(summary: dict) -> dict:
    cands = pd.read_parquet(DATA_DIR / "candidates_overnight.parquet")
    events = cands["is_event"].fillna(False).astype(bool)
    return {
        "rough_candidates": int(len(cands)),
        "events": int(events.sum()),
        "backtested": int(summary["n_trades"]),
    }


def main() -> None:
    summary = json.loads((DATA_DIR / "bt_summary.json").read_text())
    trades = pd.read_parquet(DATA_DIR / "bt_trades.parquet")
    daily = pd.read_parquet(DATA_DIR / "bt_daily.parquet")

    trades["gap_at_open"] = trades["open"] / trades["prev_close"] - 1.0

    # Equity curve, sampled to ~160 points (keeps shape)
    step = max(1, round(len(daily) / 160))
    curve = daily[["date", "cum_pnl"]].iloc[::step].copy()
    if (len(daily) - 1) % step:
        curve = pd.concat([curve, daily[["date", "cum_pnl"]].iloc[[-1]]])
    equity = {
        "dates": curve["date"].tolist(),
        "cum_pnl": [round(v) for v in curve["cum_pnl"]],
    }

    # Yearly PnL and trade counts (monthly is too dense beyond a few years)
    yg = trades.copy()
    yg["year"] = yg["date"].str[:4]
    yearly = {
        y: {"pnl": round(float(g["pnl"].sum())), "n": int(len(g)), "win": round(float((g["pnl"] > 0).mean()), 4)}
        for y, g in yg.groupby("year")
    }

    # Histogram of per-trade returns
    edges = [-np.inf, -1.0, -0.5, -0.25, -0.10, 0.0, 0.10, 0.25, 0.5, np.inf]
    labels = ["< -100%", "-100..-50%", "-50..-25%", "-25..-10%", "-10..0%", "0..10%", "10..25%", "25..50%", "> 50%"]
    counts = pd.cut(trades["ret_on_notional"], bins=edges, labels=labels).value_counts().reindex(labels)
    histogram = {"labels": labels, "counts": [int(c) for c in counts]}

    # When the overnight high printed (ET)
    t = trades.dropna(subset=["on_high_time"]).copy()
    is_prev_evening = t["on_high_time"].str[:10] != t["date"]
    hour = t["on_high_time"].str[11:13].astype(int)
    hi_when = {
        "Prior 16:00-20:00": int(is_prev_evening.sum()),
        "04:00-07:00": int(((~is_prev_evening) & (hour < 7)).sum()),
        "07:00-08:00": int(((~is_prev_evening) & (hour == 7)).sum()),
        "08:00-09:00": int(((~is_prev_evening) & (hour == 8)).sum()),
        "09:00-09:30": int(((~is_prev_evening) & (hour == 9)).sum()),
    }

    by_gap = bucket_stats(
        trades, "gap_at_open",
        bins=[-1, 0.25, 0.5, 1.0, 2.0, 100],
        labels=["<25%", "25-50%", "50-100%", "100-200%", ">200%"],
    )
    by_pmvol = bucket_stats(
        trades, "pm_dollar_vol",
        bins=[0, 3e6, 10e6, 30e6, 1e15],
        labels=["$1-3M", "$3-10M", "$10-30M", ">$30M"],
    )

    still2x = trades[trades["open"] >= 2.0 * trades["prev_close"]]
    faded = trades[trades["open"] < 2.0 * trades["prev_close"]]
    by_open_state = [
        {
            "bucket": "Still >= 2x at open",
            "n": int(len(still2x)),
            "win_rate": round(float((still2x["ret_on_notional"] > 0).mean()), 4),
            "mean": round(float(still2x["ret_on_notional"].mean()), 4),
            "median": round(float(still2x["ret_on_notional"].median()), 4),
            "worst": round(float(still2x["ret_on_notional"].min()), 4),
            "pnl": round(float(still2x["pnl"].sum()), 0),
        },
        {
            "bucket": "Faded below 2x by open",
            "n": int(len(faded)),
            "win_rate": round(float((faded["ret_on_notional"] > 0).mean()), 4),
            "mean": round(float(faded["ret_on_notional"].mean()), 4),
            "median": round(float(faded["ret_on_notional"].median()), 4),
            "worst": round(float(faded["ret_on_notional"].min()), 4),
            "pnl": round(float(faded["pnl"].sum()), 0),
        },
    ]

    # Borrow/locate cost sensitivity: flat % of notional per trade
    total_notional = float(trades["entry_notional"].sum())
    sens = []
    for c in [0.0, 0.01, 0.02, 0.05, 0.10]:
        sens.append(
            {
                "locate_pct": c,
                "total_pnl": round(float(trades["pnl"].sum()) - c * total_notional),
                "mean_ret": round(float(trades["ret_on_notional"].mean()) - c, 4),
            },
        )
    breakeven = float(trades["ret_on_notional"].mean())

    worst = trades.nsmallest(8, "pnl")[
        ["date", "ticker", "avg_px_open", "avg_px_close", "qty", "pnl", "ret_on_notional", "gap_at_open"]
    ]

    out = {
        "summary": summary,
        "equity": equity,
        "yearly": yearly,
        "histogram": histogram,
        "hi_when": hi_when,
        "by_gap": by_gap,
        "by_pmvol": by_pmvol,
        "by_open_state": by_open_state,
        "sensitivity": sens,
        "breakeven_locate_pct": round(breakeven, 4),
        "worst_trades": [
            {
                "date": r["date"],
                "ticker": r["ticker"],
                "entry": round(float(r["avg_px_open"]), 2),
                "exit": round(float(r["avg_px_close"]), 2),
                "qty": int(r["qty"]),
                "pnl": round(float(r["pnl"])),
                "ret": round(float(r["ret_on_notional"]), 3),
                "gap_open": round(float(r["gap_at_open"]), 3),
            }
            for r in worst.to_dict("records")
        ],
        "funnel": funnel(summary),
    }
    (DATA_DIR / "canvas_data.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:2000])
    print("...")
    print(f"bytes={len(json.dumps(out))}")


if __name__ == "__main__":
    main()

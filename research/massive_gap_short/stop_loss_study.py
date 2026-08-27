"""
Does a fixed intraday stop loss help the gap short overall?

For every PIT trade (bt_trades_pit.parquet) and stop level X, simulate a
buy-stop at entry * (1 + X) on the held minute bars:

  * trigger: first held bar whose high touches the stop
  * fill conventions:
      TOUCH  = max(stop, bar open)          - stop honored unless the bar
               gapped over it (e.g. a LULD reopen prints straight through)
      MINUTE = max(TOUCH, bar close)        - pessimistic: also eats any
               within-minute momentum after the touch
  * no re-entry; untriggered trades exit exactly as in the PIT run
  * a halt that never printed at/above the stop before the tape stopped is
    NOT protected: the first print after resumption is the fill for stop and
    no-stop alike (stop-market and PIT cover meet at the same print)

"Overall" is judged on three layers (flat $10k/trade, entry-date attribution,
same conventions as backtest_pit.py):
  1. per-trade edge: mean / median / p05 / worst, whipsaw cost on stopped trades
  2. per-day Kelly f* on daily close PnL
  3. the binding constraint: worst intraday portfolio trough -> the max f that
     holds the historical worst dip to -50% of equity, and compounded
     growth/year at that f (MINUTE fills)

Usage: python stop_loss_study.py
Outputs: data/stop_study.parquet + printed comparison table.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_pit as bp


DATA_DIR = Path(__file__).resolve().parent / "data"
UNIT = 10_000.0
STOPS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00, 2.00, 3.00, 4.00, 5.00]
EQUITY_DIP_FLOOR = 0.50  # size so the worst historical intraday dip is -50% of equity


def load_trade(row: dict) -> dict | None:
    """Reload held bars for one PIT trade and precompute stop triggers."""
    bars = bp.minute_bars(row["ticker"], row["prev_date"], row["date"], row["date"])
    if not len(bars):
        return None
    hold = bars[(bars["end_min"] > row["entry_min"]) & (bars["end_min"] <= row["exit_min"])]
    entry = row["entry_px"]
    com_frac = row["commission"] / (row["qty"] * row["entry_raw"])
    com_frac_sameday = 0.01 / row["entry_raw"]  # $0.005/share each way on raw shares
    out = {
        "date": row["date"],
        "ticker": row["ticker"],
        "ret_orig": row["ret_on_notional"],
        "exit_kind": row["exit_kind"],
        "entry": entry,
        "com_frac": com_frac,
        "mark_min": hold["end_min"].to_numpy(dtype=int) if len(hold) else np.array([row["entry_min"]], dtype=int),
        "mark_pnl": (
            UNIT * ((entry - hold["c"].to_numpy(dtype=float)) / entry - com_frac)
            if len(hold)
            else np.array([UNIT * -com_frac])
        ),
    }
    if len(hold):
        cm_high = np.maximum.accumulate(hold["h"].to_numpy(dtype=float))
        opens = hold["o"].to_numpy(dtype=float)
        closes = hold["c"].to_numpy(dtype=float)
        for x in STOPS:
            stop_px = entry * (1.0 + x)
            i = int(np.searchsorted(cm_high, stop_px, side="left"))
            if i >= len(cm_high):
                out[f"trig_{x}"] = -1
                continue
            touch = max(stop_px, opens[i])
            minute = max(touch, closes[i])
            out[f"trig_{x}"] = i
            out[f"ret_touch_{x}"] = (entry - touch) / entry - com_frac_sameday
            out[f"ret_minute_{x}"] = (entry - minute) / entry - com_frac_sameday
    else:
        for x in STOPS:
            out[f"trig_{x}"] = -1
    return out


def daily_trough(trades: list[dict], ret_key: str | None, trig_key: str | None) -> pd.DataFrame:
    """Per-day close PnL and minute-synchronous trough for one stop variant."""
    by_date: dict[str, list[dict]] = {}
    for t in trades:
        by_date.setdefault(t["date"], []).append(t)
    rows = []
    for date in sorted(by_date):
        day = by_date[date]
        grid = np.unique(np.concatenate([t["mark_min"] for t in day]))
        total = np.zeros(len(grid))
        close = 0.0
        for t in day:
            i = t[trig_key] if trig_key else -1
            if i >= 0:
                final = UNIT * t[ret_key]
                marks = np.concatenate([t["mark_pnl"][:i], [final]])
                mins = np.concatenate([t["mark_min"][:i], [t["mark_min"][i]]])
            else:
                final = UNIT * t["ret_orig"]
                marks, mins = t["mark_pnl"], t["mark_min"]
            idx = np.searchsorted(mins, grid, side="right") - 1
            total += np.where(idx >= 0, marks[np.clip(idx, 0, None)], 0.0)
            close += final
        rows.append({"date": date, "pnl": close, "trough": min(float(total.min()), close)})
    return pd.DataFrame(rows)


def evaluate(trades: list[dict], label: str, x: float | None, years: float) -> dict:
    if x is None:
        rets = np.array([t["ret_orig"] for t in trades])
        ret_touch = rets
        trig_key, ret_key = None, None
        n_stopped = 0
        whipsaw = np.nan
    else:
        trig_key, ret_key = f"trig_{x}", f"ret_minute_{x}"
        stopped = [t for t in trades if t[trig_key] >= 0]
        n_stopped = len(stopped)
        rets = np.array([t[ret_key] if t[trig_key] >= 0 else t["ret_orig"] for t in trades])
        ret_touch = np.array([t[f"ret_touch_{x}"] if t[trig_key] >= 0 else t["ret_orig"] for t in trades])
        # whipsaw: what the stopped trades would have made if held (PIT exit)
        whipsaw = float(np.mean([t[ret_key] - t["ret_orig"] for t in stopped])) if stopped else np.nan

    daily = daily_trough(trades, ret_key, trig_key)
    units = daily["pnl"].to_numpy() / UNIT
    trough_units = daily["trough"].to_numpy() / UNIT
    sharpe_active = float(units.mean() / units.std(ddof=1) * np.sqrt(252))
    cal = pd.Series(0.0, index=pd.bdate_range(daily["date"].min(), daily["date"].max()))
    cal[pd.to_datetime(daily["date"])] = units
    sharpe_cal = float(cal.mean() / cal.std(ddof=1) * np.sqrt(252))
    f_kelly = bp.kelly_opt(units)
    f_safe = EQUITY_DIP_FLOOR / abs(trough_units.min())
    f_use = min(f_kelly / 2, f_safe)  # half Kelly, capped by intraday survival
    growth = float(np.log1p(np.clip(f_use * units, -0.999, None)).sum() / years)
    return {
        "stop": label,
        "pct_stopped": n_stopped / len(trades),
        "mean_touch": float(ret_touch.mean()),
        "mean_minute": float(rets.mean()),
        "median": float(np.median(rets)),
        "win": float((rets > 0).mean()),
        "p05": float(np.quantile(rets, 0.05)),
        "worst": float(rets.min()),
        "whipsaw_per_stopped": whipsaw,
        "sharpe_active": sharpe_active,
        "sharpe_cal": sharpe_cal,
        "kelly_f_day": f_kelly,
        "worst_trough_units": float(trough_units.min()),
        "f_safe_50pct": f_safe,
        "f_used": f_use,
        "growth_yr_at_f": float(np.expm1(growth)),
    }


def main() -> None:
    t = pd.read_parquet(DATA_DIR / "bt_trades_pit.parquet")
    cands = pd.read_parquet(DATA_DIR / "candidates_overnight.parquet")[["date", "ticker", "prev_date"]]
    t = t.merge(cands, on=["date", "ticker"], how="left", validate="one_to_one")
    print(f"PIT trades: {len(t)}")

    rows = t.to_dict("records")
    trades = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i, r in enumerate(pool.map(load_trade, rows), 1):
            if r is not None:
                trades.append(r)
            if i % 500 == 0:
                print(f"  [{i}/{len(rows)}]", flush=True)
    print(f"loaded: {len(trades)}")

    d0 = pd.to_datetime(t["date"].min())
    d1 = pd.to_datetime(t["date"].max())
    years = (d1 - d0).days / 365.25

    results = [evaluate(trades, "none", None, years)]
    results += [evaluate(trades, f"{int(x*100)}%", x, years) for x in STOPS]

    df = pd.DataFrame(results)
    df.to_parquet(DATA_DIR / "stop_study.parquet", index=False)

    pd.set_option("display.width", 250)
    fmt = {
        "pct_stopped": "{:.1%}", "mean_touch": "{:+.2%}", "mean_minute": "{:+.2%}",
        "median": "{:+.2%}", "win": "{:.1%}", "p05": "{:+.2%}", "worst": "{:+.2%}",
        "whipsaw_per_stopped": "{:+.2%}", "sharpe_active": "{:.2f}", "sharpe_cal": "{:.2f}",
        "kelly_f_day": "{:.4f}",
        "worst_trough_units": "{:+.2f}", "f_safe_50pct": "{:.4f}", "f_used": "{:.4f}",
        "growth_yr_at_f": "{:+.1%}",
    }
    out = df.copy()
    for c, f in fmt.items():
        out[c] = out[c].map(lambda v, f=f: f.format(v) if pd.notna(v) else "-")
    print("\nMINUTE fills (pessimistic) unless noted; growth at min(half Kelly, f_safe):")
    print(out.to_string(index=False))

    # color: what the chosen level does to the known disasters and whipsaws
    x = 0.25
    print(f"\nExamples at the {int(x*100)}% stop (MINUTE fills):")
    for tk, dt in [("QMMM", "2025-09-09"), ("SPRB", "2025-10-06"), ("PAVS", "2026-06-09"), ("SPI", "2020-09-23")]:
        tr = next((z for z in trades if z["ticker"] == tk and z["date"] == dt), None)
        if tr is None:
            continue
        trig = tr[f"trig_{x}"]
        new = tr[f"ret_minute_{x}"] if trig >= 0 else tr["ret_orig"]
        print(f"  {dt} {tk:5s} held {tr['ret_orig']:+8.1%}  ->  stop {new:+8.1%}"
              f"  ({'triggered' if trig >= 0 else 'never triggered (' + tr['exit_kind'] + ')'})")

    # disaster-stop levels trigger on so few trades that listing them all is the analysis
    for x in [z for z in STOPS if z >= 2.0]:
        hit = [t for t in trades if t[f"trig_{x}"] >= 0]
        print(f"\nTriggered at the {int(x*100)}% stop ({len(hit)} trades):")
        for t in sorted(hit, key=lambda z: z[f"ret_minute_{x}"]):
            print(f"  {t['date']} {t['ticker']:5s} held {t['ret_orig']:+8.1%}  ->  stop {t[f'ret_minute_{x}']:+8.1%}")

    (DATA_DIR / "stop_study_meta.json").write_text(json.dumps({"years": years, "n_trades": len(trades)}, indent=2))


if __name__ == "__main__":
    main()

"""
Point-in-time (PIT) backtest of the overnight gap short, correcting the two
lookahead defects found in the audit of backtest_nautilus.py, and adding
intraday mark-to-market statistics.

Fix 1 - no hindsight event filters:
  * The full-day filters (n_rth_bars >= 60, first bar <= 10:30, last bar
    >= 15:59) are removed; none of them is knowable at entry time.
  * Names that halt intraday and never resume before the session end are NOT
    dropped: the short holds overnight and covers at the close of the first
    RTH minute bar on the next day the ticker prints (resumption). If the
    ticker never prints again within RESUME_MAX_SESSIONS, the trade is marked
    at the last pre-halt print and flagged `unresolved_halt`.
  * Half-days (13:00 closes) are traded with the exit trigger at 12:58 and
    the entry cutoff at 12:00 (session_end - 2min / - 60min, same offsets as
    normal days). Half-day dates are detected cross-sectionally from the last
    print time across all candidates on the date (the early-close calendar is
    known ex ante in reality).

Fix 2 - time-of-trade (raw) prices for decisions and costs:
  * Aggregates in the cache are split-adjusted as of fetch time; whether a
    trade passes a price floor or how many shares $10k buys must not depend
    on later splits. A per-(date, ticker) factor from raw grouped dailies
    (adjusted=false) rescales adjusted prices to as-traded prices.
  * The $1.00 price floor applies to the RAW official open.
  * qty = int($10k / raw entry fill); commissions $0.005/share on raw share
    counts (exit-leg share count rescaled if a split occurred during a halt).
  * Returns are computed in adjusted space (scale-consistent across halts and
    splits); commissions are converted to a fraction of true notional.

Fill convention (identical to the engine backtest, validated on the overlap):
  * Entry: market order reacting after the open; fills at the close of the
    first RTH minute bar (the last print before 09:31 in the normal case).
    Entry abandoned if the first print lands after the entry cutoff.
  * Exit: fills at the close of the first bar ending at/after the exit
    trigger (last print before ~15:58, or ~12:58 on half-days).

Intraday statistics (flat $10k per trade, net of commissions):
  * Per-trade max adverse excursion (MAE) from minute-bar highs while held.
  * Per-day portfolio trough: minute-synchronous marks (bar closes) summed
    across open positions; resumption trades are marked at the last print
    until day end and their final PnL is included in the day close.
  * Max drawdown of the cumulative equity path including intraday troughs.
  * Daily PnL is attributed to the ENTRY date (resumption losses land on the
    day the cluster was traded - the conservative choice for per-day Kelly).

Usage: python backtest_pit.py [YEAR]   (YEAR limits events for a smoke run)

Outputs: data/bt_trades_pit.parquet, data/bt_daily_pit.parquet,
data/bt_summary_pit.json.
"""

from __future__ import annotations

import gzip
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import massive_api as m


DATA_DIR = Path(__file__).resolve().parent / "data"
ET = "America/New_York"

NOTIONAL = 10_000.0
COMMISSION_PER_SHARE = 0.005
MIN_RAW_OPEN = 1.0
MIN_PM_DOLLAR_VOL = 1_000_000.0
RESUME_MAX_SESSIONS = 30
UNIT = 10_000.0  # flat sizing for daily stats, matches bt_results.py

_grouped_cache: dict[str, dict[str, dict]] = {}
_grouped_lock = threading.Lock()


def trading_dates() -> list[str]:
    return sorted(p.name.split(".")[0] for p in (DATA_DIR / "grouped_daily").glob("*.json.gz"))


def grouped_day(date: str) -> dict[str, dict]:
    """Adjusted grouped-daily rows for one date, keyed by ticker (in-process memo)."""
    with _grouped_lock:
        if date in _grouped_cache:
            return _grouped_cache[date]
    path = DATA_DIR / "grouped_daily" / f"{date}.json.gz"
    if not path.exists():
        rows: dict[str, dict] = {}
    else:
        with gzip.open(path, "rt") as f:
            data = json.load(f)
        rows = {r["T"]: r for r in data.get("results") or []}
    with _grouped_lock:
        _grouped_cache[date] = rows
    return rows


def detect_half_days(cands: pd.DataFrame) -> set[str]:
    """
    Early-close (13:00) dates, detected from the latest print across ALL
    candidates on the date. The real early-close calendar is known ex ante;
    this just recovers it from the data without a calendar dependency.
    """
    c = cands.dropna(subset=["rth_last_ts"]).copy()
    c["last_t"] = c["rth_last_ts"].str[11:16]
    last_by_date = c.groupby("date")["last_t"].max()
    return set(last_by_date[(last_by_date >= "12:45") & (last_by_date <= "13:05")].index)


def build_factors(dates: list[str], needed: dict[str, set[str]]) -> tuple[dict[tuple[str, str], float], int]:
    """
    (date, ticker) -> raw/adjusted price factor from grouped dailies.
    Fetches raw grouped dailies (cached on disk) for every date first.
    """
    print(f"ensuring raw grouped dailies for {len(dates)} dates ...", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(m.grouped_daily_raw, dates))

    factors: dict[tuple[str, str], float] = {}
    missing = 0
    for d in dates:
        adj = grouped_day(d)
        raw = {r["T"]: r for r in (m.grouped_daily_raw(d).get("results") or [])}
        for t in needed[d]:
            a, r = adj.get(t), raw.get(t)
            if a and r and a.get("c") and r.get("c"):
                factors[(d, t)] = float(r["c"]) / float(a["c"])
            else:
                missing += 1
    return factors, missing


_factor_lazy: dict[tuple[str, str], float] = {}


def lazy_factor(date: str, ticker: str, default: float) -> float:
    """Raw/adjusted factor for a date not prefetched (resumption exits)."""
    key = (date, ticker)
    if key in _factor_lazy:
        return _factor_lazy[key]
    adj = grouped_day(date).get(ticker)
    raw = next((x for x in (m.grouped_daily_raw(date).get("results") or []) if x["T"] == ticker), None)
    f = default
    if adj and raw and adj.get("c") and raw.get("c"):
        f = float(raw["c"]) / float(adj["c"])
    _factor_lazy[key] = f
    return f


def minute_bars(ticker: str, date_from: str, date_to: str, day: str) -> pd.DataFrame:
    """Event-day RTH minute bars (bar start in [09:30, 16:00) ET), from the disk cache."""
    data = m.minute_aggs(ticker, date_from, date_to)
    results = data.get("results") or []
    if not results:
        return pd.DataFrame()
    t0 = pd.Timestamp(f"{day} 09:30", tz=ET).value // 1_000_000
    t1 = pd.Timestamp(f"{day} 16:00", tz=ET).value // 1_000_000
    df = pd.DataFrame(results)
    df = df[(df["t"] >= t0) & (df["t"] < t1) & (df["c"] > 0)].sort_values("t").reset_index(drop=True)
    if not len(df):
        return df
    end = pd.to_datetime(df["t"] + 60_000, unit="ms", utc=True).dt.tz_convert(ET)
    df["end_hm"] = end.dt.strftime("%H:%M")
    df["end_min"] = end.dt.hour * 60 + end.dt.minute
    return df


def resumption_exit(ticker: str, date: str, all_dates: list[str]) -> tuple[str, float, str] | None:
    """
    First exit opportunity after an unresolved end-of-day halt: the close of
    the first RTH minute bar on the next date the ticker prints (fallback to
    that date's official open). Returns (exit_date, adjusted_px, kind).
    """
    i = all_dates.index(date)
    for d in all_dates[i + 1 : i + 1 + RESUME_MAX_SESSIONS]:
        row = grouped_day(d).get(ticker)
        if row is None:
            continue
        bars = minute_bars(ticker, d, d, d)
        if len(bars):
            return d, float(bars.iloc[0]["c"]), "resumption"
        if row.get("o"):
            return d, float(row["o"]), "resumption_open"
    return None


def simulate(row: dict, half_days: set[str], factors: dict, all_dates: list[str]) -> dict:
    date, ticker = row["date"], row["ticker"]
    out = {
        "date": date,
        "ticker": ticker,
        "prev_close": row["prev_close"],
        "open": row["open"],
        "on_high": row["on_high"],
        "pm_dollar_vol": row["pm_dollar_vol"],
        "factor": factors.get((date, ticker), np.nan),
        "traded": False,
        "skip_reason": None,
    }
    factor = out["factor"]
    if not np.isfinite(factor) or factor <= 0:
        factor = 1.0
        out["factor_fallback"] = True
    else:
        out["factor_fallback"] = False
    out["raw_open"] = row["open"] * factor

    is_half = date in half_days
    exit_trigger = "12:58" if is_half else "15:58"
    entry_cutoff = "12:00" if is_half else "15:00"
    out["half_day"] = is_half

    bars = minute_bars(ticker, row["prev_date"], date, date)
    if not len(bars):
        out["skip_reason"] = "no_rth_prints"
        return out

    entry_bar = bars.iloc[0]
    if entry_bar["end_hm"] > entry_cutoff:
        out["skip_reason"] = "first_print_after_cutoff"
        return out

    entry_px = float(entry_bar["c"])
    entry_raw = entry_px * factor
    qty = int(NOTIONAL / entry_raw)
    if qty < 1:
        out["skip_reason"] = "qty_lt_1"
        return out

    exit_candidates = bars[bars["end_hm"] >= exit_trigger]
    unresolved = False
    if len(exit_candidates):
        exit_bar = exit_candidates.iloc[0]
        exit_px = float(exit_bar["c"])
        exit_date, exit_kind, exit_hm = date, "same_day", str(exit_bar["end_hm"])
        hold = bars[(bars["end_min"] > int(entry_bar["end_min"])) & (bars["end_min"] <= int(exit_bar["end_min"]))]
    else:
        res = resumption_exit(ticker, date, all_dates)
        hold = bars[bars["end_min"] > int(entry_bar["end_min"])]
        if res is None:
            exit_px = float(bars.iloc[-1]["c"])
            exit_date, exit_kind, exit_hm = date, "unresolved_halt", str(bars.iloc[-1]["end_hm"])
            unresolved = True
        else:
            exit_date, exit_px, exit_kind = res
            exit_hm = "resume"

    if exit_date == date:
        factor_exit = factor
    else:
        factor_exit = factors.get((exit_date, ticker)) or lazy_factor(exit_date, ticker, factor)
    if not np.isfinite(factor_exit) or factor_exit <= 0:
        factor_exit = factor
    qty_exit = qty * factor / factor_exit  # share count after any split during a halt

    gross_ret = (entry_px - exit_px) / entry_px
    notional = qty * entry_raw
    commission = COMMISSION_PER_SHARE * (qty + qty_exit)
    pnl = notional * gross_ret - commission
    ret = pnl / notional
    com_frac = commission / notional

    mae_px = float(hold["h"].max()) if len(hold) else entry_px
    if exit_kind.startswith("resumption"):
        mae_px = max(mae_px, exit_px)
    mae = max(mae_px - entry_px, 0.0) / entry_px

    # Minute marks for the portfolio trough (flat $10k, commissions deducted).
    if len(hold):
        mark_min = hold["end_min"].to_numpy(dtype=int)
        mark_pnl = UNIT * ((entry_px - hold["c"].to_numpy(dtype=float)) / entry_px - com_frac)
    else:
        mark_min = np.array([int(entry_bar["end_min"])], dtype=int)
        mark_pnl = np.array([UNIT * -com_frac])

    out.update(
        traded=True,
        entry_et=str(entry_bar["end_hm"]),
        exit_et=exit_hm,
        exit_date=exit_date,
        exit_kind=exit_kind,
        unresolved_halt=unresolved,
        entry_px=entry_px,
        exit_px=exit_px,
        entry_raw=entry_raw,
        qty=qty,
        commission=commission,
        gross_ret=gross_ret,
        ret_on_notional=ret,
        pnl_10k=UNIT * ret,
        mae=mae,
        entry_min=int(entry_bar["end_min"]),
        exit_min=int(mark_min[-1]),
        mark_min=mark_min,
        mark_pnl=mark_pnl,
    )
    return out


def daily_intraday(trades: list[dict]) -> pd.DataFrame:
    """Per-day close PnL, minute-synchronous trough and peak (flat $10k/trade)."""
    by_date: dict[str, list[dict]] = {}
    for t in trades:
        by_date.setdefault(t["date"], []).append(t)

    rows = []
    for date in sorted(by_date):
        day = by_date[date]
        grid = np.unique(np.concatenate([t["mark_min"] for t in day]))
        total = np.zeros(len(grid))
        for t in day:
            # forward-fill each position's last mark across the shared grid;
            # same-day exits end on their realized PnL, resumption trades stay
            # at the last pre-halt mark (their final PnL enters only the close)
            idx = np.searchsorted(t["mark_min"], grid, side="right") - 1
            series = np.where(idx >= 0, t["mark_pnl"][np.clip(idx, 0, None)], 0.0)
            total += series
        close = float(sum(t["pnl_10k"] for t in day))
        rows.append(
            {
                "date": date,
                "n": len(day),
                "pnl": close,
                "trough": float(min(total.min(), close)),
                "peak": float(max(total.max(), close)),
            },
        )
    return pd.DataFrame(rows)


def kelly_opt(returns: np.ndarray) -> float:
    if returns.min() >= 0:
        return float("nan")  # no losing day in sample; Kelly unbounded
    f_max = 0.999 / abs(returns.min())
    grid = np.linspace(1e-4, f_max, 20_000)
    growth = [(np.log1p(f * returns)).mean() for f in grid]
    return float(grid[int(np.argmax(growth))])


def validate_vs_engine(t: pd.DataFrame) -> dict:
    """Fill-convention regression test vs the engine backtest on the overlap."""
    old = pd.read_parquet(DATA_DIR / "bt_trades.parquet")
    j = t[t["exit_kind"] == "same_day"].merge(
        old[["date", "ticker", "avg_px_open", "avg_px_close"]], on=["date", "ticker"], how="inner",
    )
    if not len(j):
        return {"overlap": 0}
    e = (j["entry_px"] - j["avg_px_open"]).abs() / j["avg_px_open"]
    x = (j["exit_px"] - j["avg_px_close"]).abs() / j["avg_px_close"]
    return {
        "overlap": int(len(j)),
        "entry_px_match_1bp": float((e < 1e-4).mean()),
        "exit_px_match_1bp": float((x < 1e-4).mean()),
        "entry_px_max_rel_err": float(e.max()),
        "exit_px_max_rel_err": float(x.max()),
    }


def main() -> None:
    year = sys.argv[1] if len(sys.argv) > 1 else None

    cands = pd.read_parquet(DATA_DIR / "candidates_overnight.parquet")
    half_days = detect_half_days(cands)
    print(f"half-day sessions detected: {len(half_days)}")

    ev = cands[cands["is_event"].fillna(False).astype(bool)].copy()
    ev = ev[ev["pm_dollar_vol"] >= MIN_PM_DOLLAR_VOL]
    if year:
        ev = ev[ev["date"].str[:4] == year]
    print(f"events (is_event & pm$vol>=1M{f' & {year}' if year else ''}): {len(ev)}")

    dates = sorted(ev["date"].unique())
    needed: dict[str, set[str]] = {}
    for d, tk in zip(ev["date"], ev["ticker"]):
        needed.setdefault(d, set()).add(tk)
    factors, n_missing = build_factors(dates, needed)
    print(f"factors built: {len(factors)} (missing -> fallback 1.0: {n_missing})")

    all_dates = trading_dates()
    rows = ev.to_dict("records")
    results = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = [pool.submit(simulate, r, half_days, factors, all_dates) for r in rows]
        for i, f in enumerate(futs, 1):
            results.append(f.result())
            if i % 500 == 0:
                print(f"  [{i}/{len(rows)}]", flush=True)

    res = pd.DataFrame([{k: v for k, v in r.items() if k not in ("mark_min", "mark_pnl")} for r in results])

    # Universe accounting: the raw $1 floor is applied HERE so both sides are measurable.
    res["passes_floor"] = res["raw_open"] >= MIN_RAW_OPEN
    traded_mask = res["traded"] & res["passes_floor"]
    sub1 = res[res["traded"] & ~res["passes_floor"]]
    trades_df = res[traded_mask].copy()
    trade_dicts = [
        r for r in results
        if r["traded"] and r["raw_open"] >= MIN_RAW_OPEN
    ]

    print("\n--- universe accounting ---")
    print(f"simulatable events:            {int(res['traded'].sum())}")
    print(f"excluded by raw $1 floor:      {len(sub1)}  (their mean ret {sub1['ret_on_notional'].mean():+.4%})")
    print(f"traded (PIT universe):         {len(trades_df)}")
    for reason, n in res.loc[~res["traded"], "skip_reason"].value_counts().items():
        print(f"  not tradeable - {reason}: {n}")
    print(f"resumption exits: {int((trades_df['exit_kind'].str.startswith('resumption')).sum())}"
          f"  unresolved halts: {int(trades_df['unresolved_halt'].sum())}"
          f"  half-day trades: {int(trades_df['half_day'].sum())}")

    daily = daily_intraday(trade_dicts)
    daily["cum"] = daily["pnl"].cumsum()
    cum_prev = daily["cum"] - daily["pnl"]
    running_peak = np.maximum.accumulate(cum_prev + daily["peak"])
    daily["dd_intraday"] = cum_prev + daily["trough"] - running_peak
    peak_close = daily["cum"].cummax()
    daily["dd_close"] = daily["cum"] - peak_close

    r = trades_df["ret_on_notional"]
    cal = pd.date_range(trades_df["date"].min(), trades_df["date"].max(), freq="B").strftime("%Y-%m-%d")
    pnl_cal = daily.set_index("date")["pnl"].reindex(cal).fillna(0.0)
    daily_units = daily.set_index("date")["pnl"] / UNIT

    summary = {
        "n_trades": int(len(trades_df)),
        "n_days": int(len(daily)),
        "n_tickers": int(trades_df["ticker"].nunique()),
        "date_range": [trades_df["date"].min(), trades_df["date"].max()],
        "win_rate": float((r > 0).mean()),
        "mean_ret": float(r.mean()),
        "median_ret": float(r.median()),
        "std_ret": float(r.std()),
        "p05_ret": float(r.quantile(0.05)),
        "p95_ret": float(r.quantile(0.95)),
        "worst_ret": float(r.min()),
        "total_pnl_10k": float(trades_df["pnl_10k"].sum()),
        "total_commission_frac_mean": float((trades_df["commission"] / (trades_df["qty"] * trades_df["entry_raw"])).mean()),
        "sharpe_daily_ann_active": float(daily["pnl"].mean() / daily["pnl"].std() * np.sqrt(252)),
        "sharpe_daily_ann_calendar": float(pnl_cal.mean() / pnl_cal.std() * np.sqrt(252)),
        "pct_days_positive": float((daily["pnl"] > 0).mean()),
        "kelly_f_day": float(kelly_opt(daily_units.to_numpy())),
        # intraday MTM stats (flat $10k/trade)
        "mae_mean": float(trades_df["mae"].mean()),
        "mae_median": float(trades_df["mae"].median()),
        "mae_p95": float(trades_df["mae"].quantile(0.95)),
        "mae_worst": float(trades_df["mae"].max()),
        "pct_trades_mae_gt_50pct": float((trades_df["mae"] > 0.5).mean()),
        "worst_day_close": float(daily["pnl"].min()),
        "worst_day_trough": float(daily["trough"].min()),
        "max_dd_close": float(daily["dd_close"].min()),
        "max_dd_intraday": float(daily["dd_intraday"].min()),
        # sample accounting
        "n_resumption_exits": int(trades_df["exit_kind"].str.startswith("resumption").sum()),
        "n_unresolved_halts": int(trades_df["unresolved_halt"].sum()),
        "n_half_day_trades": int(trades_df["half_day"].sum()),
        "n_excluded_sub_dollar_raw": int(len(sub1)),
        "n_factor_fallback": int(trades_df["factor_fallback"].sum()),
        "validation_vs_engine": validate_vs_engine(trades_df),
    }

    trades_df.drop(columns=["skip_reason"]).to_parquet(DATA_DIR / "bt_trades_pit.parquet", index=False)
    daily.to_parquet(DATA_DIR / "bt_daily_pit.parquet", index=False)
    (DATA_DIR / "bt_summary_pit.json").write_text(json.dumps(summary, indent=2))

    print("\n" + json.dumps(summary, indent=2))
    print("\nWorst 5 trades (net, on notional):")
    cols = ["date", "ticker", "entry_px", "exit_px", "exit_date", "exit_kind", "ret_on_notional", "mae"]
    print(trades_df.nsmallest(5, "ret_on_notional")[cols].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print("\nWorst 5 intraday trough days (flat $10k/trade):")
    print(daily.nsmallest(5, "trough")[["date", "n", "pnl", "trough"]].to_string(index=False, float_format=lambda x: f"{x:,.0f}"))


if __name__ == "__main__":
    main()

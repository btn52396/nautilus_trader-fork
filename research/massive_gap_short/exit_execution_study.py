"""
Best execution for the stop exit: what actually gets us out of a squeeze?

The stop's failure mode is the LULD halt: 31% of stop triggers see the tape
halt within 5 minutes, and the worst fills (-101%) are triggers that ARE
reopen bars (the name halted below the level and reopened above it). No
resting order exits through a halt - but LULD bands are computable in real
time, so an executor can see a halt coming and act before it.

Variants simulated on every final-config trade (minute bars, pessimistic
fills, same conventions as stop_loss_study):

  V0  stop-market at L = 1.5x entry (spec)
        fill = max(L, trigger-bar open, trigger-bar close)
  V1  stop -> marketable limit capped at L x 1.03
        as V0 when the V0 fill is within the cap; else rests at the cap:
        fills on the first later bar whose low <= cap, otherwise the trade
        is UNPROTECTED (rides to the normal exit like an unstopped trade)
  V2  stop -> passive EOD-style ladder (2 min: last price, +1%, then market)
        fill approximated bar-by-bar; halts during the ladder fill at the
        pessimistic reopen price
  V3  LULD pre-emption, unconditional: exit when a bar high reaches
        0.98x the upper band (imminent halt) or L, whichever is first
  V3b LULD pre-emption, armed only near the stop: fires when the bar high
        is >= 0.85 x L AND >= 0.98 x band; else identical to V0

Band reconstruction: reference = mean of the prior 5 minute-bar closes
(official LULD ref is the 5-min mean of eligible trades, updated on 1%
moves); band = 10% for raw ref >= $3, else 20%, doubled after 15:35 ET.
Raw tier via the trade's split factor. This is an approximation - the
quote-level pass (exit_execution_quotes.py) validates the disaster subset.

Usage: python exit_execution_study.py
Output: data/exit_execution_study.parquet + printed comparison.
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
STOP_X = 0.50
CAP = 1.03            # V1 marketable-limit cap over the level
ARM = 0.85            # V3b arms when high >= ARM * L
BAND_TRIG = 0.98      # fire at 98% of the upper band
EQUITY_DIP_FLOOR = 0.50
F_PROD = 0.137


def upper_band(closes: np.ndarray, j: int, factor: float, end_min: int) -> float:
    """Approx LULD upper band for bar j from prior closes (adjusted space)."""
    lo = max(0, j - 5)
    ref = closes[lo:j].mean() if j > lo else closes[0]
    pct = 0.10 if ref * factor >= 3.0 else 0.20
    if end_min >= 15 * 60 + 35:
        pct *= 2
    return ref * (1.0 + pct)


def simulate(row: dict) -> dict | None:
    bars = bp.minute_bars(row["ticker"], row["prev_date"], row["date"], row["date"])
    if not len(bars):
        return None
    hold = bars[(bars["end_min"] > row["entry_min"]) & (bars["end_min"] <= row["exit_min"])].reset_index(drop=True)
    entry = row["entry_px"]
    L = entry * (1.0 + STOP_X)
    com = 0.01 / row["entry_raw"]
    factor = row["factor"] if np.isfinite(row["factor"]) and row["factor"] > 0 else 1.0

    out = {
        "date": row["date"], "ticker": row["ticker"],
        "ret_orig": row["ret_on_notional"], "exit_kind": row["exit_kind"],
        "pm_dollar_vol": row["pm_dollar_vol"],
        "entry_min": row["entry_min"],
    }
    if not len(hold):
        for v in ("v0", "v1", "v2", "v3", "v3b", "v3c", "v3d"):
            out[f"ret_{v}"], out[f"trig_{v}"] = row["ret_on_notional"], -1
        out["mark_min"] = np.array([row["entry_min"]], dtype=int)
        out["mark_pnl"] = np.array([UNIT * -com])
        out["halt_class"] = False
        return out

    o = hold["o"].to_numpy(float)
    h = hold["h"].to_numpy(float)
    lo_ = hold["l"].to_numpy(float)
    c = hold["c"].to_numpy(float)
    mins = hold["end_min"].to_numpy(int)
    n = len(hold)
    gap_after = np.append(np.diff(mins) > 3, row["exit_kind"] != "same_day")

    out["mark_min"] = mins
    out["mark_pnl"] = UNIT * ((entry - c) / entry - com)

    def net(fill: float) -> float:
        return (entry - fill) / entry - com

    # --- V0: spec stop-market ---
    cm_high = np.maximum.accumulate(h)
    i0 = int(np.searchsorted(cm_high, L, side="left"))
    if i0 < n:
        fill0 = max(L, o[i0], c[i0])
        out["ret_v0"], out["trig_v0"] = net(fill0), i0
        out["v0_fill_x"] = fill0 / L
    else:
        out["ret_v0"], out["trig_v0"] = row["ret_on_notional"], -1
        out["v0_fill_x"] = np.nan
    # halt-class: the V0 fill gapped >5% through the level
    out["halt_class"] = bool(i0 < n and max(L, o[i0], c[i0]) > L * 1.05)

    # --- V1: marketable limit capped at L*CAP ---
    if i0 < n:
        fill0 = max(L, o[i0], c[i0])
        if fill0 <= L * CAP:
            out["ret_v1"], out["trig_v1"] = net(fill0), i0
        else:
            j = next((k for k in range(i0 + 1, n) if lo_[k] <= L * CAP), None)
            if j is not None:
                out["ret_v1"], out["trig_v1"] = net(L * CAP), j
            else:
                out["ret_v1"], out["trig_v1"] = row["ret_on_notional"], -1  # UNPROTECTED
                out["v1_stranded"] = True
    else:
        out["ret_v1"], out["trig_v1"] = row["ret_on_notional"], -1
    out.setdefault("v1_stranded", False)

    # --- V2: passive 2-min ladder from the touch ---
    if i0 < n:
        filled = False
        for k, mult in ((1, 1.0), (2, 1.01)):
            j = i0 + k
            if j >= n:
                break
            if gap_after[j - 1]:  # halted while resting: reopen fill, pessimistic
                out["ret_v2"], out["trig_v2"] = net(max(o[j], c[j], L)), j
                filled = True
                break
            if lo_[j] <= c[i0] * mult:
                out["ret_v2"], out["trig_v2"] = net(c[i0] * mult), j
                filled = True
                break
        if not filled:
            j = min(i0 + 3, n - 1)
            out["ret_v2"], out["trig_v2"] = net(max(o[j], c[j])), j
    else:
        out["ret_v2"], out["trig_v2"] = row["ret_on_notional"], -1

    # --- V3 variants: LULD pre-emption at different arming thresholds ---
    for name, armed in (("v3", 0.0), ("v3b", ARM * L), ("v3c", 0.90 * L), ("v3d", 0.95 * L)):
        trig, fill, pre = -1, None, False
        for j in range(n):
            band = upper_band(c, j, factor, int(mins[j]))
            trig_level = max(BAND_TRIG * band, armed)
            if h[j] >= L:                      # stop level first
                trig, fill = j, max(L, o[j], c[j])
                break
            if h[j] >= trig_level:             # imminent-halt pre-emption
                trig, fill, pre = j, max(trig_level, o[j], c[j]), True
                break
        if trig >= 0:
            out[f"ret_{name}"], out[f"trig_{name}"] = net(fill), trig
            out[f"pre_{name}"] = pre
            out[f"fill_{name}"] = fill
        else:
            out[f"ret_{name}"], out[f"trig_{name}"] = row["ret_on_notional"], -1
            out[f"pre_{name}"] = False
            out[f"fill_{name}"] = np.nan
        # was there a halt right after (or on) the fire bar?
        out[f"haltnext_{name}"] = bool(trig >= 0 and gap_after[min(trig, n - 1)])
        out[f"trig_min_{name}"] = int(mins[trig]) if trig >= 0 else -1

    # pre-halt actionability: did the variant exit before the first halt
    # that follows the V0 trigger (if any)?
    if i0 < n:
        halt_j = next((k for k in range(i0, n) if gap_after[k]), None)
        out["halt_after_trig"] = halt_j is not None
    else:
        out["halt_after_trig"] = False
    return out


def evaluate(trades: list[dict], key: str, label: str, years: float) -> dict:
    rets = np.array([t[f"ret_{key}"] for t in trades])  # falls back to ret_orig when unfired
    fired = np.array([t[f"trig_{key}"] >= 0 for t in trades])
    daily = sls.daily_trough(trades, f"ret_{key}", f"trig_{key}")
    units = daily["pnl"].to_numpy() / UNIT
    troughs = daily["trough"].to_numpy() / UNIT
    f_kelly = bp.kelly_opt(units)
    f_safe = EQUITY_DIP_FLOOR / abs(troughs.min())
    f_use = min(f_kelly / 2, f_safe)
    growth = float(np.expm1(np.log1p(np.clip(f_use * units, -0.999, None)).sum() / years))
    hc = [t for t in trades if t["halt_class"]]
    hc_rets = np.array([t[f"ret_{key}"] for t in hc]) if hc else np.array([0.0])
    return {
        "variant": label,
        "fired": float(fired.mean()),
        "mean": float(rets.mean()),
        "median": float(np.median(rets)),
        "win": float((rets > 0).mean()),
        "p05": float(np.quantile(rets, 0.05)),
        "worst": float(rets.min()),
        "haltclass_mean": float(hc_rets.mean()),
        "haltclass_worst": float(hc_rets.min()),
        "worst_trough": float(troughs.min()),
        "f_use": f_use,
        "growth_at_f_use": growth,
        "growth_at_f137": float(np.expm1(np.log1p(np.clip(F_PROD * units, -0.999, None)).sum() / years)),
        "growth_at_f194": float(np.expm1(np.log1p(np.clip(0.194 * units, -0.999, None)).sum() / years)),
    }


def main() -> None:
    t = pd.read_parquet(DATA_DIR / "bt_trades_pit.parquet")
    c = pd.read_parquet(DATA_DIR / "candidates.parquet")[["date", "ticker", "open_ratio"]]
    co = pd.read_parquet(DATA_DIR / "candidates_overnight.parquet")[["date", "ticker", "prev_date"]]
    t = t.merge(c, on=["date", "ticker"], how="left").merge(co, on=["date", "ticker"], how="left")
    t = t[t["open_ratio"] >= 1.15]
    t = t.sort_values("pm_dollar_vol", ascending=False).groupby("date").head(5)
    print(f"final-config trades: {len(t)}")

    rows = t.to_dict("records")
    trades = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i, r in enumerate(pool.map(simulate, rows), 1):
            if r is not None:
                trades.append(r)
            if i % 500 == 0:
                print(f"  [{i}/{len(rows)}]", flush=True)
    print(f"loaded: {len(trades)}")

    d0, d1 = t["date"].min(), t["date"].max()
    years = (pd.to_datetime(d1) - pd.to_datetime(d0)).days / 365.25

    n_halt = sum(1 for x in trades if x["halt_class"])
    n_strand = sum(1 for x in trades if x.get("v1_stranded"))
    print(f"halt-class trades (V0 fill >5% through level): {n_halt}")
    print(f"V1 stranded (cap never refilled, unprotected): {n_strand}")

    results = [
        evaluate(trades, "v0", "V0 stop-market (spec)", years),
        evaluate(trades, "v1", f"V1 capped limit L+{CAP - 1:.0%}", years),
        evaluate(trades, "v2", "V2 passive 2-min ladder", years),
        evaluate(trades, "v3", "V3 band pre-empt, always", years),
        evaluate(trades, "v3b", f"V3b band pre-empt, armed >={ARM:.0%}L", years),
        evaluate(trades, "v3c", "V3c band pre-empt, armed >=90%L", years),
        evaluate(trades, "v3d", "V3d band pre-empt, armed >=95%L", years),
    ]
    df = pd.DataFrame(results)
    df.to_parquet(DATA_DIR / "exit_execution_study.parquet", index=False)
    flat = pd.DataFrame([{k: v for k, v in x.items() if k not in ("mark_min", "mark_pnl")} for x in trades])
    flat.to_parquet(DATA_DIR / "exit_execution_trades.parquet", index=False)

    pd.set_option("display.width", 250)
    fmt = {"fired": "{:.1%}", "mean": "{:+.2%}", "median": "{:+.2%}", "win": "{:.1%}",
           "p05": "{:+.2%}", "worst": "{:+.2%}", "haltclass_mean": "{:+.2%}",
           "haltclass_worst": "{:+.2%}", "worst_trough": "{:+.2f}", "f_use": "{:.3f}",
           "growth_at_f_use": "{:+.1%}", "growth_at_f137": "{:+.1%}", "growth_at_f194": "{:+.1%}"}
    out = df.copy()
    for col, f in fmt.items():
        out[col] = out[col].map(lambda v, f=f: f.format(v) if pd.notna(v) else "-")
    print("\nExit execution variants (pessimistic minute-bar fills):")
    print(out.to_string(index=False))

    # fire decomposition + split-half stability for the pre-emption variants
    mid = sorted(x["date"] for x in trades)[len(trades) // 2]
    for key in ("v3b", "v3c", "v3d"):
        pre = [x for x in trades if x[f"pre_{key}"]]
        pre_halt = sum(1 for x in pre if x[f"haltnext_{key}"])
        would_stop = sum(1 for x in pre if x["trig_v0"] >= 0)
        print(f"\n{key}: {len(pre)} band-fires ({len(pre) / len(trades):.1%}); "
              f"{pre_halt} ({pre_halt / max(len(pre), 1):.0%}) halted on/after the fire bar; "
              f"{would_stop} would have hit L anyway under V0")
        dr = np.array([x[f"ret_{key}"] - x["ret_v0"] for x in pre])
        print(f"  band-fire delta vs V0: mean {dr.mean():+.2%} | median {np.median(dr):+.2%} "
              f"| p10 {np.quantile(dr, .1):+.2%} | p90 {np.quantile(dr, .9):+.2%}")
    print("\nsplit-half growth at own f_use:")
    for label, sub in (("H1", [x for x in trades if x["date"] < mid]),
                       ("H2", [x for x in trades if x["date"] >= mid])):
        yrs = years / 2
        r = {k: evaluate(sub, k, k, yrs) for k in ("v0", "v3c", "v3d")}
        print(f"  {label}: " + " | ".join(
            f"{k} {r[k]['growth_at_f_use']:+.0%} (f {r[k]['f_use']:.3f}, mean {r[k]['mean']:+.2%}, "
            f"trough {r[k]['worst_trough']:+.2f})" for k in r))


if __name__ == "__main__":
    main()

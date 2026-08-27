"""
Fetch extended-hours minute bars for each rough candidate and compute overnight
gap statistics.

Overnight window: previous trading day 16:00 ET (using the official previous
close as reference) through 09:30 ET on the event day. Note: on the few
half-days per year the 13:00-16:00 post-close window is not scanned.

Outputs candidates_overnight.parquet with one row per candidate including
`is_event` = overnight_high >= 2x prev_close.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import massive_api as m


DATA_DIR = Path(__file__).resolve().parent / "data"
ET = "America/New_York"
GAP_MULTIPLE = 2.0  # 100%+ overnight gap


def analyze(row: dict) -> dict:
    out = dict(row)
    try:
        data = m.minute_aggs(row["ticker"], row["prev_date"], row["date"])
    except Exception as e:  # noqa: BLE001 - record and continue
        out["error"] = str(e)[:200]
        return out

    results = data.get("results") or []
    out["error"] = None
    if not results:
        out["n_bars"] = 0
        return out

    df = pd.DataFrame(results)
    ts = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(ET)
    df["ts_et"] = ts
    out["n_bars"] = len(df)

    prev_16 = pd.Timestamp(f"{row['prev_date']} 16:00", tz=ET)
    day_930 = pd.Timestamp(f"{row['date']} 09:30", tz=ET)
    day_1600 = pd.Timestamp(f"{row['date']} 16:00", tz=ET)
    day_400 = pd.Timestamp(f"{row['date']} 04:00", tz=ET)

    on = df[(df["ts_et"] >= prev_16) & (df["ts_et"] < day_930)]
    pm = df[(df["ts_et"] >= day_400) & (df["ts_et"] < day_930)]
    rth = df[(df["ts_et"] >= day_930) & (df["ts_et"] < day_1600)]

    pc = row["prev_close"]
    if len(on):
        i_hi = on["h"].idxmax()
        out["on_high"] = float(on["h"].max())
        out["on_high_time"] = str(on.loc[i_hi, "ts_et"])
        out["on_dollar_vol"] = float((on["v"] * on["vw"]).sum())
        out["on_last_price"] = float(on.iloc[-1]["c"])
    else:
        out["on_high"] = float("nan")
        out["on_high_time"] = None
        out["on_dollar_vol"] = 0.0
        out["on_last_price"] = float("nan")

    out["pm_dollar_vol"] = float((pm["v"] * pm["vw"]).sum()) if len(pm) else 0.0
    out["n_rth_bars"] = int(len(rth))
    if len(rth):
        out["rth_first_ts"] = str(rth.iloc[0]["ts_et"])
        out["rth_last_ts"] = str(rth.iloc[-1]["ts_et"])
        out["rth_first_open"] = float(rth.iloc[0]["o"])
        # High-water mark of intraday price *after* the open, for squeeze stats
        out["rth_high"] = float(rth["h"].max())
        out["rth_low"] = float(rth["l"].min())
    else:
        out["rth_first_ts"] = None
        out["rth_last_ts"] = None
        out["rth_first_open"] = float("nan")
        out["rth_high"] = float("nan")
        out["rth_low"] = float("nan")

    out["on_gap_max"] = out["on_high"] / pc - 1.0 if out["on_high"] == out["on_high"] else float("nan")
    out["is_event"] = bool(out["on_high"] >= GAP_MULTIPLE * pc) if out["on_high"] == out["on_high"] else False
    return out


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    cands = pd.read_parquet(DATA_DIR / "candidates.parquet")
    rows = cands.to_dict("records")
    print(f"Analyzing {len(rows)} candidates with {workers} workers", flush=True)

    out_rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, res in enumerate(pool.map(analyze, rows), 1):
            out_rows.append(res)
            if i % 500 == 0:
                n_ev = sum(1 for r in out_rows if r.get("is_event"))
                print(f"  [{i}/{len(rows)}] events so far: {n_ev}", flush=True)

    df = pd.DataFrame(out_rows)
    df.to_parquet(DATA_DIR / "candidates_overnight.parquet", index=False)
    n_err = df["error"].notna().sum() if "error" in df else 0
    print(f"Done. events={int(df['is_event'].sum())} errors={n_err}")


if __name__ == "__main__":
    main()

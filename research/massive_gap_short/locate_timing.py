"""
Locate-timing analysis: when does the 100%+ gap signal first trigger, and how
much of the final premarket dollar volume (the 0.5% participation-cap input)
is visible at various premarket checkpoints?

Uses the disk-cached Massive minute aggregates for every traded backtest event.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import massive_api as m

DATA_DIR = Path(__file__).resolve().parent / "data"
ET = "America/New_York"

CHECKPOINTS = ["07:00", "08:00", "08:30", "09:00", "09:15", "09:25"]


def analyze(row: dict) -> dict | None:
    try:
        data = m.minute_aggs(row["ticker"], row["prev_date"], row["date"])
    except Exception:
        return None
    results = data.get("results") or []
    if not results:
        return None

    df = pd.DataFrame(results)
    df["ts_et"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(ET)

    prev_16 = pd.Timestamp(f"{row['prev_date']} 16:00", tz=ET)
    day_400 = pd.Timestamp(f"{row['date']} 04:00", tz=ET)
    day_930 = pd.Timestamp(f"{row['date']} 09:30", tz=ET)

    on = df[(df["ts_et"] >= prev_16) & (df["ts_et"] < day_930)]
    if not len(on):
        return None

    # First minute where the 2x threshold is crossed
    trig = on[on["h"] >= 2.0 * row["prev_close"]]
    if not len(trig):
        return None
    trig_ts = trig.iloc[0]["ts_et"]

    pm = df[(df["ts_et"] >= day_400) & (df["ts_et"] < day_930)].copy()
    pm["dv"] = pm["v"] * pm["vw"]
    final = pm["dv"].sum()
    if final <= 0:
        return None

    out = {
        "ticker": row["ticker"],
        "date": row["date"],
        "trig_ts": str(trig_ts),
        "trig_hm": trig_ts.strftime("%H:%M") if trig_ts >= day_400 else "prior_ah",
        "final_pm_dv": final,
    }
    cum_at_trig = pm.loc[pm["ts_et"] <= trig_ts, "dv"].sum()
    out["frac_at_trigger"] = cum_at_trig / final
    for hm in CHECKPOINTS:
        cutoff = pd.Timestamp(f"{row['date']} {hm}", tz=ET)
        out[f"cum_{hm}"] = pm.loc[pm["ts_et"] < cutoff, "dv"].sum()
    return out


def main() -> None:
    ev = pd.read_parquet(DATA_DIR / "bt_events.parquet")
    rows = ev.to_dict("records")
    print(f"analyzing {len(rows)} events")

    out = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i, res in enumerate(pool.map(analyze, rows), 1):
            if res:
                out.append(res)
            if i % 1000 == 0:
                print(f"  {i}/{len(rows)}")

    df = pd.DataFrame(out)
    df.to_parquet(DATA_DIR / "locate_timing.parquet", index=False)
    n = len(df)
    print(f"\nevents with trigger found: {n}")

    # --- when does the signal fire? ---
    def bucket(hm: str) -> str:
        if hm == "prior_ah":
            return "prior-day 16:00-20:00"
        if hm < "07:00":
            return "04:00-07:00"
        if hm < "08:00":
            return "07:00-08:00"
        if hm < "08:30":
            return "08:00-08:30"
        if hm < "09:00":
            return "08:30-09:00"
        return "09:00-09:30"

    df["trig_bucket"] = df["trig_hm"].map(bucket)
    order = ["prior-day 16:00-20:00", "04:00-07:00", "07:00-08:00", "08:00-08:30", "08:30-09:00", "09:00-09:30"]
    print("\nWhen the 2x gap first triggers:")
    counts = df["trig_bucket"].value_counts().reindex(order).fillna(0).astype(int)
    for b, c in counts.items():
        print(f"  {b:24s} {c:5d}  ({c/n:5.1%})")

    print(f"\nPremarket $vol visible at trigger: median {df['frac_at_trigger'].median():.1%}, "
          f"p25 {df['frac_at_trigger'].quantile(.25):.1%}, p75 {df['frac_at_trigger'].quantile(.75):.1%}")

    # --- how much of final pm $vol is known at each checkpoint? ---
    print("\nCheckpoint: fraction of final pm $vol | $1M-filter agreement | log-log R^2 vs final")
    final = df["final_pm_dv"]
    passes_final = final >= 1_000_000  # all should, by construction of the tradeable set
    for hm in CHECKPOINTS:
        cum = df[f"cum_{hm}"]
        frac = cum / final
        # would the >=$1M filter (scaled by median visibility) agree with the final decision?
        mask = cum > 0
        r2 = np.corrcoef(np.log(cum[mask & (cum > 0)]), np.log(final[mask & (cum > 0)]))[0, 1] ** 2
        agree_raw = ((cum >= 1_000_000) == passes_final).mean()
        print(f"  {hm}  median frac {frac.median():5.1%}  p25 {frac.quantile(.25):5.1%}  p75 {frac.quantile(.75):5.1%}"
              f"  | raw >=$1M agrees {agree_raw:5.1%} | R2 {r2:.2f}")

    # --- forecast multiplier: final = cum * k ---
    print("\nForecast multiplier k = final / cum (median, IQR):")
    for hm in ["08:00", "08:30", "09:00", "09:15"]:
        k = final / df[f"cum_{hm}"].replace(0, np.nan)
        print(f"  {hm}: median {k.median():5.2f}  IQR [{k.quantile(.25):5.2f}, {k.quantile(.75):5.2f}]"
              f"  p90 {k.quantile(.90):5.2f}")

    # sizing error if cap set from forecast = cum * median_k at 08:30 vs true cap
    k830 = (final / df["cum_08:30"].replace(0, np.nan)).median()
    fc = df["cum_08:30"] * k830
    err = fc / final
    print(f"\nIf sized at 08:30 with median multiplier {k830:.2f}: forecast/true cap ratio "
          f"median {err.median():.2f}, p10 {err.quantile(.10):.2f}, p90 {err.quantile(.90):.2f}")


if __name__ == "__main__":
    main()

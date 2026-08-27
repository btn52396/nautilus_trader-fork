"""
Measure opening/closing auction sizes for gap-short events via the Massive
trades endpoint (condition 16 = official open print, 15 = official close),
and compare them to the 0.5% x premarket-volume participation cap.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import massive_api as m

DATA_DIR = Path(__file__).resolve().parent / "data"
ET = "America/New_York"

F = 0.0529
PARTICIPATION = 0.005


def cross_print(ticker: str, day: str, t_from: str, t_to: str, cond: int) -> tuple[float, float] | None:
    t0 = int(pd.Timestamp(f"{day} {t_from}", tz=ET).value)
    t1 = int(pd.Timestamp(f"{day} {t_to}", tz=ET).value)
    try:
        data = m._get(
            f"/v3/trades/{ticker}",
            {"timestamp.gte": t0, "timestamp.lte": t1, "limit": 5000, "order": "asc"},
        )
    except Exception:
        return None
    for r in data.get("results") or []:
        conds = r.get("conditions") or []
        if cond in conds:
            return float(r["price"]), float(r["size"])
    return None


def analyze(row: dict) -> dict:
    out = {
        "ticker": row["ticker"],
        "date": row["date"],
        "pm_dollar_vol": row["pm_dollar_vol"],
        "open_cross_usd": np.nan,
        "close_cross_usd": np.nan,
    }
    oc = cross_print(row["ticker"], row["date"], "09:29:59", "09:32:00", 16)
    if oc is None:
        oc = cross_print(row["ticker"], row["date"], "09:32:00", "10:30:00", 16)  # delayed opens
    if oc:
        out["open_cross_usd"] = oc[0] * oc[1]
    cc = cross_print(row["ticker"], row["date"], "15:59:58", "16:00:30", 15)
    if cc:
        out["close_cross_usd"] = cc[0] * cc[1]
    return out


def main() -> None:
    t = pd.read_parquet(DATA_DIR / "bt_trades.parquet")
    sub = t[t["date"] >= "2026-01-01"].copy()
    print(f"measuring auctions for {len(sub)} events (2026)")

    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, res in enumerate(pool.map(analyze, sub.to_dict("records")), 1):
            rows.append(res)
            if i % 100 == 0:
                print(f"  {i}/{len(sub)}")

    df = pd.DataFrame(rows)
    df.to_parquet(DATA_DIR / "auction_sizes.parquet", index=False)

    ok_o = df["open_cross_usd"].notna()
    ok_c = df["close_cross_usd"].notna()
    print(f"\nopen cross found: {ok_o.mean():.0%}  close cross found: {ok_c.mean():.0%}")

    d = df[ok_o & ok_c].copy()
    d["open_ratio"] = d["open_cross_usd"] / d["pm_dollar_vol"]
    d["close_ratio"] = d["close_cross_usd"] / d["pm_dollar_vol"]

    print("\nAuction size distribution ($):")
    for col in ["open_cross_usd", "close_cross_usd"]:
        q = d[col].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        print(f"  {col:16s} p10 ${q[0.1]:>10,.0f}  p25 ${q[0.25]:>10,.0f}  median ${q[0.5]:>10,.0f}"
              f"  p75 ${q[0.75]:>10,.0f}  p90 ${q[0.9]:>10,.0f}")

    print("\nAuction as % of premarket $vol:")
    for col in ["open_ratio", "close_ratio"]:
        q = d[col].quantile([0.25, 0.5, 0.75])
        print(f"  {col:12s} p25 {q[0.25]:6.2%}  median {q[0.5]:6.2%}  p75 {q[0.75]:6.2%}")

    print("\nOur size as % of the OPEN cross (MOO) at various equity levels:")
    for eq in [40_000, 100_000, 250_000, 500_000, 1_000_000]:
        size = np.minimum(F * eq, PARTICIPATION * d["pm_dollar_vol"])
        pct = size / d["open_cross_usd"]
        print(f"  equity ${eq:>9,}: median {pct.median():6.1%}  p75 {pct.quantile(.75):6.1%}"
              f"  p90 {pct.quantile(.90):6.1%}  | >20% of cross on {(pct > 0.20).mean():5.1%} of events")

    print("\nSame vs the CLOSE cross (MOC):")
    for eq in [40_000, 100_000, 250_000, 500_000, 1_000_000]:
        size = np.minimum(F * eq, PARTICIPATION * d["pm_dollar_vol"])
        pct = size / d["close_cross_usd"]
        print(f"  equity ${eq:>9,}: median {pct.median():6.1%}  p75 {pct.quantile(.75):6.1%}"
              f"  p90 {pct.quantile(.90):6.1%}  | >20% of cross on {(pct > 0.20).mean():5.1%} of events")


if __name__ == "__main__":
    main()

"""
Test the proposed entry: sell-short limit placed at the start of the 09:31 bar
at that bar's open price, resting until filled or 15:00 ET. Exit at the
engine's 15:58 fill (avg_px_close). Fill requires the tape to trade up through
the limit; three queue-buffer assumptions are reported.

Baseline for comparison: the engine's marketable 09:31 entry (fills 100%).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import massive_api as m

ET = "America/New_York"
ENTRY_CUTOFF = "15:00"


def analyze(row: dict) -> dict | None:
    try:
        d = m.minute_aggs(row["ticker"], row["date"], row["date"])
    except Exception:
        return None
    res = d.get("results") or []
    if not res:
        return None
    df = pd.DataFrame(res)
    df["et"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(ET)
    df["hm"] = df["et"].dt.strftime("%H:%M")
    rth = df[(df["hm"] >= "09:31") & (df["hm"] <= ENTRY_CUTOFF)]
    if not len(rth):
        return None

    limit_px = float(rth.iloc[0]["o"])  # price at placement (~09:31:00)
    out = {
        "date": row["date"],
        "ticker": row["ticker"],
        "limit_px": limit_px,
        "engine_ret": row["ret_on_notional"],
        "exit_px": row["avg_px_close"],
    }
    for eps, name in [(1.0, "fill_t_strict"), (1.001, "fill_t_e1"), (1.002, "fill_t_e2")]:
        thresh = limit_px * eps
        hit = rth[rth["h"] > thresh] if eps == 1.0 else rth[rth["h"] >= thresh]
        out[name] = hit.iloc[0]["hm"] if len(hit) else None
    return out


def main() -> None:
    t = pd.read_parquet("data/bt_trades.parquet")
    rows = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for r in pool.map(analyze, t.to_dict("records")):
            if r:
                rows.append(r)
    d = pd.DataFrame(rows)
    n = len(d)
    print(f"events: {n}")
    print(f"baseline engine (marketable 09:31): mean {d['engine_ret'].mean():+.2%}, fills 100%")
    print()

    for col, label in [
        ("fill_t_strict", "strict: trades > limit"),
        ("fill_t_e1", "queue buffer 0.1%"),
        ("fill_t_e2", "queue buffer 0.2%"),
    ]:
        filled = d[col].notna()
        com = 0.01 / d["limit_px"]  # $0.005/share each way as return fraction
        ret_filled = (d["limit_px"] - d["exit_px"]) / d["limit_px"] - com
        per_event = np.where(filled, ret_filled, 0.0)
        missed = d[~filled]
        late = d[filled & (d[col] > "09:35")]
        print(f"[{label}]")
        print(f"  fill rate {filled.mean():.1%}  | mean/filled {ret_filled[filled].mean():+.2%}"
              f"  | mean/event {per_event.mean():+.2%}  vs engine {d['engine_ret'].mean():+.2%}")
        print(f"  missed {len(missed)} trades; their engine returns: mean {missed['engine_ret'].mean():+.2%}"
              f"  median {missed['engine_ret'].median():+.2%}  total@$10k ${(missed['engine_ret'] * 10_000).sum():,.0f}")
        print(f"  fills after 09:35: {len(late)} ({len(late)/max(filled.sum(),1):.1%} of fills), "
              f"their mean ret {ret_filled[filled & (d[col] > '09:35')].mean():+.2%}")
        print()

    # Fill-delay distribution for the middle assumption
    ft = d["fill_t_e1"].dropna()
    print("fill-time distribution (0.1% buffer):")
    for cut in ["09:31", "09:32", "09:35", "09:45", "10:30", "12:00", "15:00"]:
        print(f"  filled by {cut}: {(ft <= cut).mean():.1%}")


if __name__ == "__main__":
    main()

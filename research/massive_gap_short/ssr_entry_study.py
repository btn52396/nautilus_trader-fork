"""
How much does Rule 201 (SSR) actually cost the entry?

~23% of events arrive at the open already SSR-flagged (prior-day low <= 0.9x
the close two days back). The backtest fills every entry at the 09:31 print,
and the rev-1 ladder sim allowed resting AT the bid and crossing - both
executions Rule 201 forbids for a short on a flagged name.

The counter-hypothesis (plausible): pegged at bid + 1 tick, the order fills
on the next uptick anyway, so the haircut is ~zero.

Measurement: for a sample of SSR-flagged PIT trades, replay the real quote
and trade tape 09:31-09:35 twice on identical data:

  FREE ladder (rev-1): mid -> bid over 10 steps of 10s, marketable at the
      bid allowed; passive fill when a print >= the resting limit
  SSR ladder: identical rungs floored at bid + 1 tick, crossing disallowed,
      re-pegged as the bid moves; fill when a print >= the limit (any such
      print is above the bid -> Rule 201 compliant)

Both fills are compared to each other and to the 09:31 print (the PIT
convention). Unfilled-by-09:35 SSR orders are marked and valued at the final
bid + 1 tick (optimistic; the unfilled rate is reported alongside).

Usage: python ssr_entry_study.py [n_sample]
Output: data/ssr_entry_study.parquet + printed summary.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_pit as bp
import test_entry_ladder as tel

ET = "America/New_York"
OUT = Path(__file__).resolve().parent / "data" / "ssr_entry_study.parquet"
N_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SEED = 7


def tick(px: float) -> float:
    return 0.01 if px >= 1.0 else 0.0001


def ssr_flags(trades: pd.DataFrame) -> pd.Series:
    """Prior-day low <= 0.9 x close two sessions back (rev-1 definition)."""
    dates = bp.trading_dates()
    idx = {d: i for i, d in enumerate(dates)}
    flags = []
    for r in trades.to_dict("records"):
        i = idx.get(r["prev_date"])
        if i is None or i == 0:
            flags.append(False)
            continue
        prev = bp.grouped_day(r["prev_date"]).get(r["ticker"])
        prev2 = bp.grouped_day(dates[i - 1]).get(r["ticker"])
        flags.append(
            prev is not None and prev2 is not None
            and prev.get("l") is not None and prev2.get("c")
            and float(prev["l"]) <= 0.9 * float(prev2["c"])
        )
    return pd.Series(flags, index=trades.index)


def simulate(row: dict) -> dict | None:
    day, tkr = row["date"], row["ticker"]
    t_open = pd.Timestamp(f"{day} 09:30:00", tz=ET).value
    t_pre = pd.Timestamp(f"{day} 09:29:00", tz=ET).value
    t_end = t_open + tel.SIM_END_S * 1_000_000_000
    t_start = t_open + tel.START_S * 1_000_000_000

    try:
        quotes = tel.fetch_all(f"/v3/quotes/{tkr}", {
            "timestamp.gte": t_pre, "timestamp.lte": t_end, "limit": 50000, "order": "asc"})
        trades = tel.fetch_all(f"/v3/trades/{tkr}", {
            "timestamp.gte": t_pre, "timestamp.lte": t_end, "limit": 50000, "order": "asc"})
    except Exception:
        return None
    if not quotes or not trades:
        return None

    q = pd.DataFrame(quotes)[["participant_timestamp", "bid_price", "ask_price"]].dropna()
    q = q[(q["bid_price"] > 0) & (q["ask_price"] >= q["bid_price"])]
    tr = pd.DataFrame(trades)[["participant_timestamp", "price", "size"]].dropna()
    if not len(q) or not len(tr):
        return None
    q_ts = q["participant_timestamp"].to_numpy()
    tr_ts = tr["participant_timestamp"].to_numpy()

    def nbbo_at(ts: int) -> tuple[float, float] | None:
        i = np.searchsorted(q_ts, ts, side="right") - 1
        if i < 0:
            return None
        r = q.iloc[i]
        return float(r["bid_price"]), float(r["ask_price"])

    i31 = np.searchsorted(tr_ts, t_start, side="right") - 1
    if i31 < 0 or nbbo_at(t_start) is None:
        return None
    px_931 = float(tr.iloc[i31]["price"])
    bid0, ask0 = nbbo_at(t_start)

    def run_ladder(ssr: bool) -> tuple[float, float, bool]:
        """Returns (fill_px, delay_s, filled_before_end)."""
        step, t_now = 0, t_start
        while t_now <= t_end:
            nb = nbbo_at(t_now)
            if nb is None:
                break
            bid, ask = nb
            mid = (bid + ask) / 2
            k = min(step, tel.N_STEPS)
            limit = mid - (k / tel.N_STEPS) * (mid - bid)
            if ssr:
                limit = max(limit, bid + tick(bid))
            elif limit <= bid:
                return limit, (t_now - t_start) / 1e9, True  # marketable
            t_next = t_now + tel.STEP_S * 1_000_000_000
            lo = np.searchsorted(tr_ts, t_now, side="left")
            hi = np.searchsorted(tr_ts, t_next, side="left")
            seg = tr.iloc[lo:hi]
            hit = seg[seg["price"] >= limit]
            if len(hit):
                return limit, (int(hit.iloc[0]["participant_timestamp"]) - t_start) / 1e9, True
            step += 1
            t_now = t_next
        nb = nbbo_at(t_end) or (bid0, ask0)
        px = nb[0] + tick(nb[0]) if ssr else nb[0]
        return px, (t_end - t_start) / 1e9, False

    free_px, free_delay, free_filled = run_ladder(ssr=False)
    ssr_px, ssr_delay, ssr_filled = run_ladder(ssr=True)

    i_next = np.searchsorted(tr_ts, t_start + 60_000_000_000, side="right") - 1
    px_next = float(tr.iloc[i_next]["price"]) if i_next >= 0 else px_931

    return {
        "date": day, "ticker": tkr,
        "px_931": px_931, "px_free": free_px, "px_ssr": ssr_px,
        "free_delay_s": free_delay, "ssr_delay_s": ssr_delay,
        "free_filled": free_filled, "ssr_filled": ssr_filled,
        "haircut_vs_free": (free_px - ssr_px) / free_px,   # >0: SSR entry worse for a short
        "haircut_vs_931": (px_931 - ssr_px) / px_931,
        "min1_up": px_next > (bid0 + ask0) / 2,
    }


def main() -> None:
    t = pd.read_parquet("data/bt_trades_pit.parquet")
    c = pd.read_parquet("data/candidates_overnight.parquet")[["date", "ticker", "prev_date"]]
    t = t.merge(c, on=["date", "ticker"], how="left", validate="one_to_one")
    t = t[ssr_flags(t)]
    print(f"SSR-flagged PIT trades: {len(t)} ({len(t)/3258:.1%} of universe)")

    sample = t.sample(min(N_SAMPLE, len(t)), random_state=SEED).to_dict("records")
    print(f"simulating {len(sample)} events (quotes+trades, 09:31-09:35)")

    rows = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        for i, r in enumerate(pool.map(simulate, sample), 1):
            if r:
                rows.append(r)
            if i % 50 == 0:
                print(f"  [{i}/{len(sample)}] ok={len(rows)}", flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(OUT, index=False)

    print(f"\nsimulated: {len(d)}")
    print(f"SSR unfilled by 09:35: {(~d['ssr_filled']).mean():.1%} (valued at final bid+tick)")
    print(f"fill delay: free median {d['free_delay_s'].median():.0f}s | ssr median {d['ssr_delay_s'].median():.0f}s")
    for col, label in [("haircut_vs_free", "SSR vs free ladder"), ("haircut_vs_931", "SSR vs 09:31 print (PIT convention)")]:
        h = d[col]
        print(f"{label}: mean {h.mean():+.2%} | median {h.median():+.2%} | p90 {h.quantile(.9):+.2%} | worst {h.max():+.2%}")
    for up, g in d.groupby("min1_up"):
        print(f"  minute-after {'UP  ' if up else 'DOWN'}: n={len(g):>3}  haircut vs 931 mean {g['haircut_vs_931'].mean():+.2%}")


if __name__ == "__main__":
    main()

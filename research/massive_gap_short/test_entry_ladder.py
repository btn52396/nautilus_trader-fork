"""
Quote-level simulation of the proposed entry ladder:

  09:30:00  sell-short limit at NBBO midpoint
  every 10s reprice 10% of the (current) mid->bid span toward the bid
  step 10 (~09:31:40) limit = bid (marketable), then re-peg to bid every 10s

Passive fill: a trade prints at >= the resting limit while it isn't crossing.
Marketable fill: limit <= current bid executes at the limit immediately.

Compared against (a) the engine baseline entry = last trade at 09:31:00 and
(b) an instant marketable order at 09:30:00 (fills at the bid). Same exit
(engine 15:58 fill) and commissions for all variants. Sample: random 2026
events, real NBBO + trades from Massive.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

import massive_api as m

ET = "America/New_York"
STEP_S = 10
N_STEPS = 10  # at the bid after 100s
START_S = 60  # ladder placement time, seconds after 09:30:00
SIM_END_S = 300
OUT_PATH = "data/entry_ladder_931.parquet"
SAVE_EVERY = 250


def fetch_all(path: str, params: dict) -> list[dict]:
    out = []
    for _ in range(6):  # max pages
        data = m._get(path, params)
        out.extend(data.get("results") or [])
        nxt = data.get("next_url")
        if not nxt:
            break
        cursor = parse_qs(urlparse(nxt).query).get("cursor", [None])[0]
        if not cursor:
            break
        params = {"cursor": cursor, "limit": params.get("limit", 50000)}
    return out


def simulate(row: dict) -> dict | None:
    day = row["date"]
    tkr = row["ticker"]
    t_open = pd.Timestamp(f"{day} 09:30:00", tz=ET).value
    t_pre = pd.Timestamp(f"{day} 09:29:00", tz=ET).value
    t_end = t_open + SIM_END_S * 1_000_000_000

    try:
        quotes = fetch_all(f"/v3/quotes/{tkr}", {
            "timestamp.gte": t_pre, "timestamp.lte": t_end, "limit": 50000, "order": "asc"})
        trades = fetch_all(f"/v3/trades/{tkr}", {
            "timestamp.gte": t_pre, "timestamp.lte": t_end, "limit": 50000, "order": "asc"})
    except Exception:
        return None
    if not quotes or not trades:
        return None

    q = pd.DataFrame(quotes)[["participant_timestamp", "bid_price", "ask_price"]].dropna()
    q = q[(q["bid_price"] > 0) & (q["ask_price"] >= q["bid_price"])]
    tr = pd.DataFrame(trades)[["participant_timestamp", "price", "size"]].dropna()
    q_ts = q["participant_timestamp"].to_numpy()
    tr_ts = tr["participant_timestamp"].to_numpy()
    if not len(q) or not len(tr):
        return None

    def nbbo_at(ts: int) -> tuple[float, float] | None:
        i = np.searchsorted(q_ts, ts, side="right") - 1
        if i < 0:
            return None
        r = q.iloc[i]
        return float(r["bid_price"]), float(r["ask_price"])

    t_start = t_open + START_S * 1_000_000_000

    # Baseline: last trade at/just before the ladder start (engine convention)
    i31 = np.searchsorted(tr_ts, t_start, side="right") - 1
    if i31 < 0:
        return None
    px_931 = float(tr.iloc[i31]["price"])

    nb0 = nbbo_at(t_start)
    if nb0 is None:
        return None
    bid0, ask0 = nb0
    px_ioc_930 = bid0  # instant marketable at ladder start
    spread0 = (ask0 - bid0) / ((ask0 + bid0) / 2)

    # Ladder simulation
    fill_px = None
    fill_t = None
    step = 0
    t_now = t_start
    while fill_px is None and t_now <= t_end:
        nb = nbbo_at(t_now)
        if nb is None:
            break
        bid, ask = nb
        mid = (bid + ask) / 2
        k = min(step, N_STEPS)
        limit = mid - (k / N_STEPS) * (mid - bid)
        if limit <= bid:
            fill_px, fill_t = limit, t_now  # crossing: execute at our limit (= bid)
            break
        # passive rest until next step: filled if a trade prints at/above limit
        t_next = t_now + STEP_S * 1_000_000_000
        lo = np.searchsorted(tr_ts, t_now, side="left")
        hi = np.searchsorted(tr_ts, t_next, side="left")
        seg = tr.iloc[lo:hi]
        hit = seg[seg["price"] >= limit]
        if len(hit):
            fill_px, fill_t = limit, int(hit.iloc[0]["participant_timestamp"])
            break
        step += 1
        t_now = t_next
    if fill_px is None:  # stale/halted tape: cross at last known bid
        nb = nbbo_at(t_end)
        if nb is None:
            return None
        fill_px, fill_t = nb[0], t_end

    exit_px = row["avg_px_close"]

    def ret(entry: float) -> float:
        return (entry - exit_px) / entry - 0.01 / entry

    # Direction of the minute after placement (for selection diagnostics)
    i_next = np.searchsorted(tr_ts, t_start + 60_000_000_000, side="right") - 1
    px_next = float(tr.iloc[i_next]["price"]) if i_next >= 0 else px_931

    return {
        "date": day,
        "ticker": tkr,
        "spread_930": spread0,
        "px_931": px_931,
        "px_ladder": fill_px,
        "px_ioc": px_ioc_930,
        "fill_delay_s": (fill_t - t_start) / 1e9,
        "ret_931": ret(px_931),
        "ret_ladder": ret(fill_px),
        "ret_ioc": ret(px_ioc_930),
        "min1_up": px_next > (bid0 + ask0) / 2,
    }


def main() -> None:
    t = pd.read_parquet("data/bt_trades.parquet")
    todo = t.to_dict("records")

    done_keys: set[tuple[str, str]] = set()
    prior: list[dict] = []
    out = Path(OUT_PATH)
    if out.exists():
        prev = pd.read_parquet(out)
        prior = prev.to_dict("records")
        done_keys = set(zip(prev["date"], prev["ticker"]))
        print(f"resuming: {len(done_keys)} events already simulated")
    todo = [r for r in todo if (r["date"], r["ticker"]) not in done_keys]
    print(f"simulating {len(todo)} events")

    rows = prior
    n_new = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, r in enumerate(pool.map(simulate, todo), 1):
            if r:
                rows.append(r)
                n_new += 1
            if i % SAVE_EVERY == 0:
                pd.DataFrame(rows).to_parquet(out, index=False)
                print(f"  {i}/{len(todo)} (saved)", flush=True)

    d = pd.DataFrame(rows)
    d.to_parquet(out, index=False)
    print(f"done: {len(d)} events simulated total ({n_new} new)")


if __name__ == "__main__":
    main()

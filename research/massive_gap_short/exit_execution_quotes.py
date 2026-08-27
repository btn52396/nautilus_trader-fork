"""
Quote-level validation of the stop's halt-gap tail (the -101% class).

For every halt-class trade (V0 fill gapped >5% through the level), replay the
real trade tape around the stop trigger and answer:

  Q1  Did prints >= L exist BEFORE the halt? If so, a live stop-market fires
      pre-halt and fills near those prints - the backtest's reopen fill is
      pessimistic. Measure the achievable fill (first print >= L, +250ms
      latency) vs the backtest convention.
  Q2  If the cross only happened inside the halt (reopen gap), confirm it:
      last pre-halt print < L, first post-halt print > L. The reopen print
      is the true fill floor - compare it to the backtest's max(L, o, c).
  Q3  LULD pre-emption actionability: from tape prints, rolling 5-min mean
      reference -> upper band; time from the 0.98-band cross to the halt.
      Is there enough tape to act on at all?

Halts are detected from the tape itself: print gaps > 120s. All price
comparisons in RAW prices (L_raw = 1.5 x entry_raw).

Usage: python exit_execution_quotes.py
Output: data/exit_execution_quotes.parquet + printed summary.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import test_entry_ladder as tel

ET = "America/New_York"
DATA_DIR = Path(__file__).resolve().parent / "data"
STOP_X = 0.50
LATENCY_NS = 250_000_000
HALT_GAP_S = 120
BAND_TRIG = 0.98


def band_pct(raw_px: float, min_of_day: int) -> float:
    pct = 0.10 if raw_px >= 3.0 else 0.20
    return pct * 2 if min_of_day >= 15 * 60 + 35 else pct


def replay(row: dict) -> dict | None:
    day, tkr = row["date"], row["ticker"]
    L_raw = row["entry_raw"] * (1.0 + STOP_X)
    t_930 = pd.Timestamp(f"{day} 09:30:00", tz=ET).value
    t_ent = t_930 + int(row["entry_min"] - 570) * 60_000_000_000
    t_eod = pd.Timestamp(f"{day} 16:05:00", tz=ET).value
    try:
        trades = tel.fetch_all(f"/v3/trades/{tkr}", {
            "timestamp.gte": t_ent - 6 * 60_000_000_000, "timestamp.lte": t_eod,
            "limit": 50000, "order": "asc"})
    except Exception:
        return None
    if not trades:
        return None
    tr = pd.DataFrame(trades)[["participant_timestamp", "price", "size"]].dropna()
    tr = tr[tr["price"] > 0].sort_values("participant_timestamp")
    ts = tr["participant_timestamp"].to_numpy(np.int64)
    px = tr["price"].to_numpy(float)
    if len(tr) < 10:
        return None

    held = ts >= t_ent
    if not held.any():
        return None
    hts, hpx = ts[held], px[held]

    # halts straight from the tape: print gaps > HALT_GAP_S
    gaps = np.where(np.diff(hts) > HALT_GAP_S * 1_000_000_000)[0]
    halt_starts = hts[gaps]              # last print before each gap

    # first cross of L on the tape
    xi = int(np.argmax(hpx >= L_raw)) if (hpx >= L_raw).any() else -1
    if xi < 0:
        return {"date": day, "ticker": tkr, "ok": False, "why": "no tape cross"}
    t_cross = int(hts[xi])
    # the cross is a reopen print if the previous print is > HALT_GAP_S older
    prev_gap_s = (t_cross - int(hts[xi - 1])) / 1e9 if xi > 0 else 0.0
    cross_is_reopen = prev_gap_s > HALT_GAP_S

    # achievable fill: first print at/after t_cross + latency
    ai = int(np.searchsorted(hts, t_cross + LATENCY_NS, side="left"))
    if ai >= len(hts):
        ai = len(hts) - 1
    fill_live = float(hpx[ai])
    fill_gap_s = (int(hts[ai]) - t_cross) / 1e9
    # did a halt begin before the achievable fill printed?
    halted_before_fill = bool(len(halt_starts) and
                              ((halt_starts >= t_cross) & (halt_starts < int(hts[ai]))).any())

    # time from cross to next halt (exit window)
    nxt = halt_starts[halt_starts >= t_cross]
    window_s = (int(nxt[0]) - t_cross) / 1e9 if len(nxt) else np.inf

    # Q3: band pre-emption from the tape (5-min rolling mean of prints, vectorized)
    seg_t, seg_p = hts[:xi + 1], hpx[:xi + 1]
    cs = np.concatenate([[0.0], np.cumsum(seg_p)])
    lo_i = np.searchsorted(seg_t, seg_t - 300_000_000_000, side="left")
    idx = np.arange(len(seg_p))
    ref = (cs[idx + 1] - cs[lo_i]) / np.maximum(idx + 1 - lo_i, 1)
    m_of_day = 570 + (seg_t - t_930) / 60_000_000_000
    pct = np.where(ref >= 3.0, 0.10, 0.20) * np.where(m_of_day >= 935, 2.0, 1.0)
    fire = seg_p >= BAND_TRIG * ref * (1.0 + pct)
    t_pre, band_x = -1.0, np.nan
    if fire.any():
        fi = int(np.argmax(fire))
        t_pre, band_x = int(seg_t[fi]), float(seg_p[fi])
    pre_lead_s = (t_cross - t_pre) / 1e9 if t_pre > 0 else np.nan
    pre_window_s = np.nan
    if t_pre > 0:
        nxt_p = halt_starts[halt_starts >= t_pre]
        pre_window_s = (int(nxt_p[0]) - t_pre) / 1e9 if len(nxt_p) else np.inf

    bt_fill_raw = row["v0_fill_x"] * L_raw
    return {
        "date": day, "ticker": tkr, "ok": True, "why": "",
        "L_raw": L_raw, "bt_fill_raw": bt_fill_raw,
        "cross_is_reopen": cross_is_reopen, "prev_gap_s": prev_gap_s,
        "fill_live_raw": fill_live, "fill_gap_s": fill_gap_s,
        "halted_before_fill": halted_before_fill, "window_s": window_s,
        "live_vs_bt": (bt_fill_raw - fill_live) / bt_fill_raw,  # >0: live better for us
        "live_x_L": fill_live / L_raw, "bt_x_L": bt_fill_raw / L_raw,
        "pre_lead_s": pre_lead_s, "pre_window_s": pre_window_s,
        "pre_px_x_L": band_x / L_raw if np.isfinite(band_x) else np.nan,
    }


def main() -> None:
    tt = pd.read_parquet(DATA_DIR / "exit_execution_trades.parquet")
    bt = pd.read_parquet(DATA_DIR / "bt_trades_pit.parquet")[
        ["date", "ticker", "entry_px", "entry_raw"]]
    hc = tt[tt["halt_class"]].merge(bt, on=["date", "ticker"], validate="one_to_one")
    print(f"halt-class trades to replay: {len(hc)}")

    out_path = DATA_DIR / "exit_execution_quotes.parquet"
    rows: list[dict] = []
    if out_path.exists():
        rows = pd.read_parquet(out_path).to_dict("records")
        done = {(r["date"], r["ticker"]) for r in rows}
        hc = hc[~hc.apply(lambda r: (r["date"], r["ticker"]) in done, axis=1)]
        print(f"resuming: {len(rows)} already replayed, {len(hc)} to go")
    if len(hc):
        with ThreadPoolExecutor(max_workers=10) as pool:
            for i, r in enumerate(pool.map(replay, hc.to_dict("records")), 1):
                if r is not None:
                    rows.append(r)
                if i % 20 == 0:
                    print(f"  [{i}/{len(hc)}]", flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(out_path, index=False)
    ok = d[d["ok"]].copy()
    for col in ("cross_is_reopen", "halted_before_fill"):
        ok[col] = ok[col].astype(bool)
    print(f"replayed: {len(d)}, usable {len(ok)}")

    ro = ok[ok["cross_is_reopen"]]
    pre = ok[~ok["cross_is_reopen"]]
    print(f"\ncross is a REOPEN print (gap through the level, no pre-halt window): "
          f"{len(ro)}/{len(ok)} ({len(ro) / len(ok):.0%})")
    print(f"cross printed pre-halt (stop-market can act): {len(pre)}")
    if len(pre):
        print(f"  exit window before next halt: median {pre['window_s'].median():.0f}s | "
              f"p10 {pre['window_s'].quantile(.1):.0f}s | halted before +250ms fill: "
              f"{pre['halted_before_fill'].mean():.0%}")
        print(f"  live fill vs backtest fill: mean {pre['live_vs_bt'].mean():+.1%} | "
              f"median {pre['live_vs_bt'].median():+.1%} | worse-than-bt share "
              f"{(pre['live_vs_bt'] < 0).mean():.0%}")
        print(f"  live fill x L: median {pre['live_x_L'].median():.2f} vs bt "
              f"{pre['bt_x_L'].median():.2f}")
    if len(ro):
        print(f"  reopen class - live fill vs backtest: mean {ro['live_vs_bt'].mean():+.1%} | "
              f"median {ro['live_vs_bt'].median():+.1%} | worse-than-bt share "
              f"{(ro['live_vs_bt'] < 0).mean():.0%}")
        print(f"  reopen fill x L: live median {ro['live_x_L'].median():.2f} vs bt "
              f"{ro['bt_x_L'].median():.2f}")

    got_pre = ok[np.isfinite(ok["pre_lead_s"]) & (ok["pre_lead_s"] > 0)]
    print(f"\nband pre-emption signal existed before the L-cross: {len(got_pre)}/{len(ok)}")
    if len(got_pre):
        print(f"  lead over the cross: median {got_pre['pre_lead_s'].median():.0f}s | "
              f"p10 {got_pre['pre_lead_s'].quantile(.1):.0f}s")
        w = got_pre["pre_window_s"]
        print(f"  tape window from signal to halt: median {w.median():.0f}s | "
              f"p10 {w.quantile(.1):.0f}s | <5s share {(w < 5).mean():.0%}")
        print(f"  signal price x L: median {got_pre['pre_px_x_L'].median():.2f}")


if __name__ == "__main__":
    main()

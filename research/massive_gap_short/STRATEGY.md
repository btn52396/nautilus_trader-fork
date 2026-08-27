# Overnight Gap Short — Full Strategy Specification

Short micro-cap stocks that have printed **2x or more above the prior regular-session close
at any point overnight**, entering just after the open with a **buy-stop at 1.5x entry**,
covering before the close. At most **5 names per day**, selected by premarket dollar volume.

This is **revision 2** of the specification, produced after a lookahead-bias audit of the
original backtest and a corrected point-in-time (PIT) rerun. The original headline
(+7.88%/trade, no stops, f = 0.1059) was measured on a universe censored of halt disasters,
with split-adjusted prices leaking future information into the $1 floor, share counts, and
commissions. The corrected pipeline (`backtest_pit.py`) fixes both, restores the halted
names with explicit resumption exits, and re-derives sizing from uncensored intraday marks.
Every parameter below is backed by a measurement on the corrected data; the evidence is in
[Appendix A](#appendix-a--evidence-behind-each-decision).

---

## 1. Headline results (corrected PIT backtest, final configuration)

Configuration measured: top-5 names/day by premarket dollar volume, 50% stop, pessimistic
stop fills (max of stop level, trigger-bar open, trigger-bar close).

| Metric | Value |
|---|---|
| Period | 2016-08-29 to 2026-08-24 (10 years) |
| Trades / active days | 3,028 / 1,399 (2.16 trades per active day) |
| Win rate | 68.1% |
| Mean return per trade (net of $0.005/sh commissions) | **+5.34%** on notional |
| Median return per trade | +11.73% |
| Trades stopped out | 16.1% |
| p05 / worst single trade | −53.8% / **−101.4%** (halt-reopen gap through the stop) |
| Daily Sharpe, annualized | 3.91 active-day / 2.83 calendar-day |
| Positive months | 78% of 115 |
| Per-day empirical Kelly (full / half) | 0.387 / 0.194 |
| Worst intraday portfolio trough (flat sizing) | −2.55x one trade's notional |

For comparison, the corrected all-qualifier universe **without** the stop measures
+6.21%/trade but troughs at **−16.05x** one trade's notional intraday (2025-09-09),
capping survivable size at f ≈ 0.031. The stop costs ~1pp of per-trade mean and multiplies
deployable size ~5x — the growth comes from the sizing, not the mean.

Locate fees and borrow interest are **not** modeled (see Sections 6 and 11).

---

## 2. What changed from revision 1

| Rev 1 | Rev 2 | Why (measured) |
|---|---|---|
| Universe required full-day RTH bars (`n_rth_bars >= 60`, `rth_last_t >= 15:59`) | No full-day filters; halted names stay in, exit at first resumption print | Lookahead: the filters deleted 59 halt events averaging −110% (QMMM −501%, SPRB −572% measured to resumption) |
| Adjusted prices for the $1 floor, shares, commissions | As-traded (raw) prices for all decisions and costs | Future splits decided sample membership; true commissions were 1.05% of notional vs 0.17% modeled |
| No stop losses ("size is the risk control") | **Buy-stop at entry x 1.5**, no re-entry | Uncensored marks show −16x intraday days; unstopped, survivable f is 0.031 and the strategy barely compounds |
| Unlimited concurrent positions (max 19/day) | **Max 5 names/day** by premarket $vol | Cluster days are correlated whipsaw days; the cap improves return AND risk (2026 max DD −74% → −49%) |
| f = 0.1059 (half Kelly on censored returns) | **f = 0.137** (see Section 5) | Kelly re-fit on corrected, stop-protected distribution with an intraday-trough constraint |
| Locate fee gate 5% of notional | **2% of notional** | Mean edge is +5.3%, not +7.9%; a 5% fee can exceed the trade's expected value |
| No explicit open-level rule (an accidental >= 1.15x funnel artifact) | **Open must hold >= 1.15x prior close** | Broadened re-scan of the missed zone (2026 pilot: 47 sub-1.15x-open events) measured no edge there (mean −0.3%, win 49%); the rule makes the candidate funnel complete and removes the last data lookahead |

---

## 3. Signal and qualification

**Watchlist trigger (loose):** any trade print >= 2.0x the prior day's regular-session
close, at any time between 16:00 ET the prior day and 09:30 ET. Add to watchlist.

- 17.9% of eventual events trigger in the prior evening session (16:00-20:00).
- 73% have triggered by 08:00 ET; only 6.7% trigger in the last half hour.
- At trigger time a median of only 3.8% of the final premarket dollar volume has printed —
  the trigger must not drive sizing.

**Qualification (all knowable before entry, nothing else):**

| Rule | Value |
|---|---|
| Cumulative premarket dollar volume (from 04:00 ET) | >= $1,000,000 |
| Price at entry, **as-traded** (unadjusted) | >= $1.00 |
| Overnight high | >= 2.0x prior regular-session close |
| Open (09:30 print) vs prior close | >= 1.15x — the gap must still be holding at the open |

The open-hold rule is measured, not assumed: a broadened re-scan of the sub-1.15x-open
zone (2026 pilot, 47 qualifying events the original funnel never fetched) found **no
edge** there — mean −0.3%, win rate 49%. A name that hit 2x overnight but faded below
1.15x by the open has already mean-reverted before it can be shorted. (Rev 1's stricter
"still >= 2x near the open" variant remains unbacktested and is not part of this spec.)

There are **no** data-quality exclusions: names that later halt, open late, or trade a
half-day session are all tradeable at 09:31 and stay in.

---

## 4. Daily selection — at most 5 names

If more than 5 names qualify, trade the **5 with the highest premarket dollar volume**.

- 94% of days have <= 5 qualifiers (44% have exactly one); the rule binds on 91 of 1,403
  active days in the decade.
- The high-count days are the correlated-squeeze days (max 15 qualifiers, 2026-06-09), and
  they were net losers: capping them improved the decade geo-mean AND cut the worst year
  drawdown. Concentration was being paid for, not compensated.
- Premarket dollar volume is also a borrow proxy: the most-traded names are the ones where
  locates exist.

---

## 5. Position sizing — f = 0.137 with a participation cap

```
target_dollars = min( 0.137 x account_equity ,           # see derivation below
                      0.005 x cum_premarket_dollar_vol ) # 0.5% participation cap
shares         = min( located_shares , target_dollars / current_price )
```

**Where 0.137 comes from.** On the corrected 50%-stop universe (all qualifiers, before the
top-5 cap), per-day Kelly is 0.301 (half = 0.151) and the worst intraday portfolio trough
is −3.64x one trade's notional, so holding the historical worst dip to −50% of equity caps
f at 0.137. The chosen constant is min(half Kelly, intraday cap) = **0.137**. Measured on
the final configuration itself (top-5, open-hold rule), the same convention would allow
f = 0.194 (Kelly 0.387, trough −2.55x), so 0.137 carries an extra safety margin against
the fit. Fit instability warning: the same constant fitted only on 2021-2026 is ~0.13, on
2016-2021 ~0.23 — do not size above what the recent regime supports.

At 5 names max, worst-case gross short exposure is ~0.69x equity, inside realistic margin
for hard-to-borrow micro-caps.

**What this sizing did historically** (replay at f = 0.137, $40k reset each January,
participation cap applied, top-5 selection, 50% stop):

| Year | End equity | Return | Worst intraday dip | Max DD |
|---|---|---|---|---|
| 2016 (from Aug 29) | $51,579 | +28.9% | −3.3% | −2.2% |
| 2017 | $41,385 | +3.5% | −8.4% | −14.3% |
| 2018 | $113,954 | +184.9% | −6.9% | −7.1% |
| 2019 | $137,386 | +243.5% | −14.7% | −18.3% |
| 2020 | $2,056,494 | +5,041.2% | −16.9% | −30.5% |
| 2021 | $370,498 | +826.2% | −35.0% | −42.3% |
| 2022 | $161,661 | +304.2% | −11.1% | −12.0% |
| 2023 | $301,802 | +654.5% | −17.6% | −26.8% |
| 2024 | $296,044 | +640.1% | −24.2% | **−57.5%** |
| 2025 | $969,560 | +2,323.9% | −21.9% | −31.6% |
| 2026 (to Aug 24) | $338,148 | +745.4% | −33.8% | −42.1% |

Geo-mean over the nine full years: **~+568%/yr**. Read the drawdown column before the
return column: the worst month is **−53% (2024-08)** and 2024 draws down −58% at this f.
Halving f to ~0.07 roughly halves every dip and drawdown figure and still compounds triple
digits in active years. The compounding ceiling is capacity, not edge: with the
participation cap the strategy stops scaling around $5-15M of equity regardless of f.

---

## 6. Locate workflow

These names are almost never easy-to-borrow — 85% of recent events had zero Interactive
Brokers shortable availability. A locate-capable broker (CenterPoint or equivalent) is a
hard requirement.

**Locate early, locate everything.** Request a locate the moment a name qualifies
(typically from ~07:00 ET). Do not wait to know the final top-5: 94% of days have <= 5
qualifiers anyway, and locating every qualifier costs only ~20 unused locates per year
versus trading. Inventory on hot gappers depletes and reprices through the morning; an
unused early locate is a small sunk fee, a missing 09:25 locate is the whole trade.

**Sizing the locate** — premarket volume is mostly invisible early, so size at checkpoints:

| Checkpoint | Median % of final premarket $vol visible | Sizing forecast multiplier |
|---|---|---|
| $1M volume cross | (qualification moment) | — |
| 08:30 ET | 72% | x1.37 |
| 09:00 ET | 88% | x1.14 |
| 09:20-09:25 ET | 98% | x1.06 |

At each checkpoint compute `target_dollars` from volume-so-far x the multiplier, round up
to the next 100-share lot, request the increment. Add ~10% buffer on cheap borrows.

**Fee gate (binary): skip the trade if the locate fee exceeds 2% of trade notional.**
IBorrowDesk data showed no fee-to-return correlation among available names — an expensive
locate is a cost, not a signal. With a +5.0% mean, a 2% fee is already 40% of the edge;
log every quote live and revisit the threshold if the paid average approaches 1%.

---

## 7. Entry — passive-to-aggressive ladder starting 09:31:00 ET

Do **not** trade the 09:30 minute (minute-one drift is adverse) and do **not** use MOO
(Section 10). At **09:31:00**:

1. Place a limit sell-short at the **midpoint** of the NBBO for
   `min(located_shares, target_dollars / current_price)` shares.
2. Every **10 seconds**, reprice 10% of the mid-to-bid distance closer to the **bid**
   (cancel/replace; nothing rests longer than one rung — nothing rests through a LULD
   halt, and you re-evaluate at any reopen).
3. At ~**09:32:40** the order sits at the bid; keep **pegging the bid every 10 seconds**
   until filled.
4. **SSR carve-out:** if the name is SSR-flagged (19.3% of PIT events arrive at the open
   already flagged), floor the ladder at **bid + 1 tick** — Rule 201 forbids short sales
   at or below the national best bid. Measured on real quote replays
   (`ssr_entry_study.py`, 300 flagged events): the uptick fill arrives as fast as the
   unconstrained ladder and the median cost is ~zero; the mean is +29bps vs the 09:31
   print, tail-driven (a collapsing bid can run from the pegged offer; worst −47% of
   entry), and 8% are still unfilled at 09:35. Blended drag on the headline mean: ~6bps.
5. If unfilled by ~15:00 ET, abandon the entry.

On the rev-1 universe the ladder measured +42bps better than crossing the spread, with 94%
of fills within 10 seconds (so the instant collapses are not missed). The backtest
convention behind every number in this document is the simpler "short the 09:31 minute
close"; treat the ladder as execution icing, not modeled edge.

---

## 8. Stop loss — buy-stop at entry x 1.50, no re-entry

Place immediately on fill: **stop-market buy at 1.5x the entry price** for the full
position. If it fires, the trade is done for the day — no re-entry.

What to expect (measured, pessimistic fills):

- **16.1% of trades trigger** (final config). 42% fill exactly at the level, two-thirds
  within 2% of it; median slippage 0.6% of the level (0.9pp of entry notional); mean
  2.6% / 3.9pp, entirely tail-driven; p95 11% / 17pp; p99 20% / 30pp and worst 34% / 51pp
  (LULD-reopen gaps through the level). Blended drag on the strategy: ~0.63pp of the
  +5.34% per-trade mean — already inside every number in this document.
- **Whipsaw is the price:** stopped trades would have made ~10pp more on average by
  holding (most squeezes fade). Every stop level lowers the per-trade mean; the stop is
  bought for the tail, not the average.
- **The tail it removes:** QMMM 2025-09-09 −501% held → −28% stopped; SPRB 2025-10-06
  −572% → −40% (at the 25% study level). Worst intraday portfolio trough −16x → −2.5x.
- **The tail it cannot remove:** a name that halts **below** the stop and reopens above it
  fills at the reopen print like any other cover (worst historical: −101%, SPRB reopening
  22% above the level after a 35-minute halt; DWACU gapped 64pp through a 25% stop). Any
  loss cap can be gapped over; size for that residual.

Why 50%: the 25-100% band all works and is robust to sizing convention and split-half
refitting; tight stops (<= 15%) whipsaw away most of the edge; disaster-only stops
(+200% to +500%) are strictly worse (they cover fade-backs at squeeze peaks while barely
truncating true runaways). 50% is mid-band: it keeps ~72% of the unstopped edge, stops few
trades, and does not ride shorts to −100% marked, where borrow recalls and margin desks
would force the cover anyway at a worse price.

### Execution of the stop: a plain stop-market, nothing cleverer

Send the full position as a **stop-market the moment the level prints. Never cap the
price, never work it passively.** Every alternative was simulated on all 3,028
final-config trades (pessimistic minute fills, same conventions as the backtest), and the
two obvious "refinements" both reintroduce the tail the stop exists to remove:

- **Capping it** (stop converts to a marketable limit at level x 1.03) saves ~6bps of
  mean and strands 6 trades whose tape gaps through the cap and never comes back: the
  limit sits unfilled and the position rides unprotected — worst trade −101% → **−572%**,
  worst intraday trough −2.55x → −6.25x of equity-at-f, compounding at the safe size
  +1,150% → +212%/yr. A cap converts "worst case: a bad fill" into "worst case: no fill".
- **Working it passively** (the 15:57-style ladder, started at the trigger) saves ~15bps
  of mean, but 31% of triggers see the name halt within five minutes, and a resting limit
  caught in that halt fills at the reopen: worst trade −253%, trough −3.11x, growth
  +789%/yr. The EOD ladder monetizes patience in a calm, scheduled state; at a stop
  trigger, patience is halt exposure.
- **LULD pre-emption** (fire early when the tape approaches the computed upper band, i.e.
  try to beat the halt) was simulated at four arming thresholds and rejected.
  Unconditionally it fires on 60% of trades and destroys the edge (+0.45% mean, 44% win).
  Armed only near the stop (>= 90% of the level) it does cut the halt-class losses
  (−66% → −47% mean) and looks better full-sample at its own safe f, but it loses to the
  plain stop in the recent half of the decade (+348% vs +470%/yr), and only 12% of its
  fires actually see a halt. On real tape the band signal precedes the level-cross by a
  median 16 minutes at 0.76x the level — it is a momentum alarm, not a halt predictor.

Reality check on the tail (all 87 worst "gapped" stop fills replayed against the real
trade tape): **72% actually printed through the level before any halt**, with a median
144-second window and a dense enough tape that an immediate market order fills in every
one — at a median 1.01x the level vs the 1.08x the backtest charges. The 28% that truly
crossed inside a halt fill at the reopen print, median 1.04x the level vs the backtest's
1.12x. Across all 82 usable replays the live-achievable fill was **never worse than the
modeled one**, so the −101% worst case and every number in this document stay
conservative under real execution.

**Triggered while halted** (name halts below the level, indicative reopen above it): the
stop becomes a market order into the reopening auction / first print. That is exactly the
modeled fill; no order type does better, and size is the only protection — which is what
f = 0.137 already prices.

---

## 9. Exit — mirror ladder starting 15:57:00 ET, hard deadline

Holding the short into the final minutes is paid (+0.37%/trade from 15:55 to 15:58). If
the stop has not fired:

1. **15:57:00** — limit buy-to-cover at the NBBO **midpoint**, full position.
2. Every **10 seconds**, reprice 10% of the mid-to-ask distance toward the **ask**.
3. ~**15:58:40** — at the ask; **peg the ask every 5 seconds**.
4. **15:59:00** — anything unfilled becomes an unconditional **market order**.

Covering is a buy: SSR never applies to the exit.

**Half-day sessions** (Jul 3, Black Friday, Christmas Eve pattern): the entire schedule
shifts to the 13:00 close — ladder at 12:57, market order 12:59. These sessions are
tradeable and averaged +6.5% in the corrected backtest.

**Halted through the close:** if the name is halted at 15:59 (LULD or regulatory) there is
no order type that exits — you hold until the tape resumes and cover at the **first print
after resumption** (first RTH minute close, or the official open if no minute bar). The
corrected backtest models exactly this: 47 such trades over the decade, mean −88%, worst
−572% unstopped / −101% with the stop having fired earlier where it could. This is the
residual tail that only position size caps.

---

## 10. Hard rules (the "never" list)

| Never | Why (measured) |
|---|---|
| No MOO / MOC orders | Median opening cross on these names is $358, median closing cross $1,309. Any real size *is* the auction. |
| No resting entry limits above the market | Misses the instant faders, which average +24% — the best trades never come back. |
| No voluntary overnight holds | The left tail is unbounded; forced holds (halts) are already the worst trades in the sample. Never add a discretionary one. |
| No re-entry after a stop-out | Not modeled; re-shorting a proven runaway re-opens the exact tail the stop just paid to close. |
| No adding to a position intraday | Not modeled, not validated. |
| No trade without a locate at <= 2% of notional | Fee is a pure cost (no fee-return correlation); above 2% it consumes ~half the mean edge. |
| No sizing above f = 0.137 | The recent-regime refit supports ~0.13; the constant is already the aggressive end. |

---

## 11. Risk profile — read before sizing

- **The stop caps most trades near −50%, not all.** Fills gap through the level on halt
  reopens: p05 is −54%, the worst fill was −101%, and a halt that starts below the stop
  gives zero protection until the reopen. Assume a few stop-outs per year land in the
  −60% to −100% range.
- **Squeeze days cluster and correlate.** The 2026 drawdown was −49% in 15 trading days:
  nine of fifteen positions stopped out on 2026-06-09 alone as the whole micro-cap complex
  squeezed together. The top-5 cap bounds this (max gross ~0.69x equity) but cannot remove
  it — on a cluster day, five positions are one trade.
- **Sequential bleed exists too.** 2024's −58% drawdown (worst month 2024-08, −53% at
  f = 0.137) was months of moderate losing days, not one event. No exposure cap fixes it;
  only f does.
- **Sharpe and win rate barely move with the stop** (3.67/67.6% vs 3.42/71.3% unstopped) —
  they are blind to the tail the stop removes. Do not evaluate this strategy on either;
  watch the intraday troughs.
- **Regime dependence.** 2020/2025 dwarf other years; 2017 made +3.5%. Event density
  (14 to 711 trades/yr) drives everything. Assume quiet regimes pay ~nothing and do not
  annualize the good years.
- **Unmodeled costs, all negative:** locate fees (gated at 2%, but every accepted fee
  subtracts directly), borrow interest for the hours held, and fill slippage at size
  beyond the participation cap's assumption. (SSR fill degradation was measured — ~6bps
  blended, tail-driven — and is no longer on this list.)
- **Kelly circularity, partially mitigated:** f was fitted on this same decade. The
  split-half check (fit one half, replay the other) supports the stop level and the
  ~0.13-0.19 f range, but a true walk-forward refit each year is the honest live protocol.
- **Funnel completeness gap: resolved, measured decade-wide.** The rev-1 candidate funnel
  could not see events that hit 2x overnight but faded below 1.15x by the open (unless
  they later squeezed intraday — a biased subset). The zone was measured twice: a 2026
  pilot (`rescan_funnel.py`, 21,134 extra pairs: 47 missed events, mean −0.3%) and a
  decade-wide estimate (`rescan_decade.py` + `zone_analysis.py`: a necessary-conditions
  screen with 100% recall on all 103 known zone events, 491k candidate pairs, 33%
  deterministic sample = 162k fetched). Decade result, PIT conventions with the 50%
  stop: ~32 zone events/yr, stratified mean **+0.55% (se 1.8%), win 46%** vs the traded
  universe's +5.34% / 68.1% — and negative net of locate fees. The never-squeezed
  sub-population alone measures +6.2%, but squeezing is unknowable at entry, and the
  squeezer stratum (census: mean −26% stopped) drags the tradeable aggregate to ~zero.
  The zone is also worst in the recent regime (2025: −1.9%, 2026: −2.6%). The spec's
  open >= 1.15x rule therefore costs ~nothing and makes the funnel complete by
  construction: every open-holding candidate was always fetched, and the 44 sub-1.15x
  trades that had leaked in via the day-high arm (mean −19%) are excluded. One residual
  sliver: the funnel's $200k day-dollar-volume garbage filter sat
  upstream of the open arm, so an open-holding name that cleared $1M premarket while
  printing < $200k all regular session could be absent. Only ~0.4/day pass even a loose
  screen (open >= 1.15x, price >= $1, RTH $100-200k), and the premarket-vs-RTH inversion
  requires an at-the-open halt — the disaster class — so any absence flatters the tail
  slightly rather than the edge. No known data lookahead remains in the pipeline for the
  specified strategy.

---

## 12. Paper-trading validation list (do these before real size)

1. **Log every locate quote** (ticker, time, fee, size available). The 2% gate and the
   fee distribution are the least-validated inputs; the running average paid is the live
   out-of-sample test of the whole cost model.
2. **Log every stop fill vs its level.** The backtest's pessimistic convention (worst of
   open/close of the trigger minute) should bound reality — the tape replay of the 87
   worst historical fills says live should beat it (median 1.01x vs 1.08x the level).
   If live slippage exceeds the convention, re-run the stop grid with the observed
   distribution.
3. Verify the SSR reject/reprice behavior of the broker/route on a flagged name.
4. Measure real ladder fill rates vs the simulated 94%-within-10s.
5. Confirm locate-at-qualification is operationally feasible from 07:00 (API, latency,
   quote expiry) and that unused locates are released or expire cheaply.
6. Dry-run the halt contingencies: (a) halted at 15:59 — you hold overnight and cover at
   the resumption print; (b) halted below the stop — the stop will fill at the reopen,
   possibly far above the level. Know both in advance; size for them.

---

## Appendix A — Evidence behind each decision

| Decision | Measurement |
|---|---|
| 2x overnight-high trigger, loose | 59,038 rough candidates → 5,120 events (overnight high >= 2x) → 3,214 tradeable after PIT gates + open-hold rule; 3,028 after the top-5 cap |
| Short works (corrected) | 68.1% win, +5.34% mean net, Sharpe 3.91 active-day, 78% positive months, positive every calendar year at chosen f |
| Open must hold >= 1.15x | Decade-wide zone measurement (three strata: day-high census + 2026 pilot census + 33% sample of a 100%-recall 491k-pair screen): ~32 sub-1.15x events/yr, weighted mean +0.55% (se 1.8%) with the 50% stop, win 46%, worst regime recently (2025 −1.9%, 2026 −2.6%) — no edge vs +5.34%/68.1% above the gate, and negative after locate fees; the rule also closes the funnel lookahead (44 day-high-arm leaks at −19% excluded) |
| 1.15x is the right level | Threshold sweep 1.10-2.00 under the final config: per-trade mean rises monotonically with the gate (+5.34% → +8.29%) but trades/yr collapse (303 → 129); growth at f and calendar Sharpe both peak at 1.15 (+578%/yr at f = 0.137, 2.83) and the ranking holds in each decade half; every band above 1.15 is positive after the stop (weakest, 1.25-1.50x: +2.30% mean), so there is nothing left to cut |
| Rev-1 numbers were inflated | Halt censoring +full-day filters: +7.88% → +6.21% mean unstopped; worst trade −434% → −572%; per-day Kelly 0.212 → 0.146; adjusted-price commissions understated true cost 0.17% → 1.05% of notional |
| 50% stop level | Grid 10-500%: tight stops whipsaw away the edge (10%: mean +0.33%); 25-100% band robust across three sizing conventions and split-half refit; 200-500% strictly worse (cover fade-backs at peaks, keep −5x to −8.5x troughs); 50% keeps 72% of edge, stops 16%, worst trough −3.64x |
| Stops help via sizing, not mean | Unstopped: f capped 0.031 by −16.05x trough, growth +82%/yr; 50% stop: f 0.137, growth +436%/yr (full universe, in-sample) despite 1.7pp lower mean |
| Top-5 cap by premarket $vol | Binds on 6% of days; cluster days were net losers: geo-mean +420% → +503%/yr AND 2026 max DD −74% → −49%; Sharpe 3.37 → 3.67 (3.91 with the open-hold rule) |
| f = 0.137 | min(half Kelly 0.151, −50%-dip cap 0.137) on the full 50%-stop universe; final config's own convention allows 0.189; recent-half refit ~0.13 |
| 0.5% participation cap | Capacity control; binds from ~$47k equity on minimum-volume names; caps strategy scaling at ~$5-15M equity |
| Locate early | 94% of days <= 5 qualifiers; on heavier days the 08:30 volume ranking matches the final five 89% (92% within top-7); locating all qualifiers costs ~20 unused locates/yr |
| Binary 2% fee gate | No fee-to-return correlation among available names (IBorrowDesk join); 85% of events zero IB availability; 2% ≈ 40% of mean edge as worst case |
| Ladder entry at 09:31 | +42bps vs crossing, 94% fills within 10s (rev-1 universe; convention icing, not modeled edge) |
| No MOO/MOC | Median opening cross $358 / closing $1,309 — no size in the crosses |
| Exit at 15:57 not 15:55 | Fade from 15:55→15:58 pays +0.37%/trade |
| Half days tradeable | 32 events at 13:00-close sessions, mean +6.5% with a 12:58 exit |
| Halt resumption exits | 47 forced overnight holds, mean −88%; worst −572% unstopped; first-print cover is the only exit that exists |
| SSR carve-out | 19.3% of PIT events arrive SSR-flagged; bid+1-tick ladder measured on 300 real quote replays: median cost ~0 vs the free ladder, mean +29bps vs the 09:31 print (tail-driven), 8% unfilled at 09:35, ~6bps blended headline drag |
| Stop exit = plain stop-market | Variant sim on all 3,028 trades: +3% price cap → worst −572%, trough −6.25x (6 stranded unfilled); passive 2-min ladder → −253%/−3.11x; LULD band pre-emption fires 60% of trades unconditionally (mean +0.45%), armed >= 90%L still loses recent-half growth (+348% vs +470%/yr) with 12% fire precision; tape replay of all 87 gapped fills: 72% printed pre-halt (median 144s window, market order fills 100%), live fill never worse than the modeled max(level, open, close) |

## Appendix B — Repo map and reproduction

All research code: `research/massive_gap_short/`. Data source: Massive REST API
(`https://api.massive.com`, key in repo `.env` as `MASSIVE_API_KEY`). Artifacts land in
`research/massive_gap_short/data/` (gitignored cache, ~GBs).

Pipeline, in order:

| Script | Produces | Purpose |
|---|---|---|
| `scan_daily.py` | `data/grouped_daily/*.json.gz` | Grouped-daily bars, all US stocks, 10y |
| `build_candidates.py` | `data/candidates.parquet` | Rough gap candidates (streamed day-pairs) |
| `fetch_overnight.py` | `data/candidates_overnight.parquet` | Extended-hours minute bars; flags `is_event` |
| `analyze.py` | `data/events_enriched.parquet` | Research stats (rev-1 filters; superseded) |
| `backtest_nautilus.py` | `data/bt_positions.csv`, `bt_fills.csv` | Rev-1 engine backtest (kept for comparison) |
| `bt_results.py` | `data/bt_trades.parquet`, `bt_summary.json` | Rev-1 joined trades (kept for comparison) |
| **`backtest_pit.py`** | `data/bt_trades_pit.parquet`, `bt_summary_pit.json` | **Corrected PIT backtest**: no hindsight filters, halt resumption exits, raw-price floor/shares/commissions, per-minute MAE and portfolio troughs |
| **`stop_loss_study.py`** | `data/stop_study.parquet` | Stop grid 10-500% on PIT trades, pessimistic fills, Sharpe/Kelly/trough/growth per level |
| **`stop_loss_robustness.py`** | `data/stop_robustness.parquet` | Stop ranking under conservative sizing + split-half refit |
| `kelly.py [F]` | stdout | Empirical Kelly + compounding replay (rev-1 universe) |
| `model_quarter_kelly.py [F]` | stdout | Rev-1 sizing replay (superseded by the Section 5 table) |
| `backtest_moo_moc.py` | stdout | Fill-convention comparison |
| `auction_capacity.py` | `data/auction_sizes.parquet` | Opening/closing cross sizes |
| `locate_timing.py` | `data/locate_timing.parquet` | Trigger times, premarket volume visibility |
| `borrow_fee_analysis.py` | `data/events_with_borrow.parquet` | IBorrowDesk fee join |
| `test_entry_ladder.py` | `data/entry_ladder_931*.parquet` | Quote-level ladder simulation |
| `ssr_entry_study.py` | `data/ssr_entry_study.parquet` | SSR (Rule 201) entry haircut: bid+1-tick ladder vs free ladder vs 09:31 print on real quotes |
| `rescan_funnel.py` | `data/rescan_funnel.parquet` | Broadened candidate re-scan measuring the sub-1.15x-open zone the rev-1 funnel missed (basis for the open-hold rule) |
| `open_threshold_study.py` | `data/open_threshold_study.parquet` | Open-gate sweep 1.10-2.00 under the final config (top-5, 50% stop): growth/Sharpe per threshold, marginal bands, split-half |
| `rescan_decade.py` | `data/rescan_decade.parquet` (+ `_pairs`) | Decade-wide sub-1.15x-zone fetch: necessary-conditions screen (100% recall on 103 knowns), deterministic extendable sampling |
| `zone_analysis.py` | `data/zone_analysis.parquet` | Zone edge, PIT-simulated with the 50% stop, Horvitz-Thompson weighted across the three strata (day-high census / pilot census / decade sample) |
| `exit_execution_study.py` | `data/exit_execution_study.parquet` (+ `_trades`) | Stop-exit execution variants on minute bars: stop-market vs capped limit vs passive ladder vs LULD band pre-emption at four arming levels; troughs, safe f, growth per variant |
| `exit_execution_quotes.py` | `data/exit_execution_quotes.parquet` | Real-tape replay of the 87 halt-class stop fills: pre-halt exit window, achievable market-order fill vs the backtest convention, band-signal lead time |

Implementation gotchas (a fresh implementation will hit them):

- **Adjusted vs raw prices.** Aggregates default to `adjusted=true`; quotes/trades are
  raw. The PIT backtest fetches `grouped_daily_raw` (unadjusted) and computes per-(date,
  ticker) factors so that the $1 floor, share counts, and commissions use as-traded
  prices while returns are computed split-safely in adjusted space. Any join across the
  two price spaces needs this rescale.
- **Mixed EST/EDT timestamps.** ET timestamp strings carry both -04:00 and -05:00 offsets;
  slice `HH:MM` from the string or parse row-wise.
- **Sub-cent commissions.** NautilusTrader `Money` (USD) is 2dp; run the engine
  commission-free and deduct `2 x qty x $0.005` post-hoc on **raw** share counts.
- **Half-day detection.** Grouped-daily minute coverage identifies 13:00-close sessions;
  the exit and stop schedule must shift with them.
- **Stop-fill modeling.** Simulate stops as: trigger on the first held minute whose high
  touches the level; fill at max(level, bar open, bar close). LULD reopens make the fill
  gap; do not model stops as fills at the level. The convention is tape-validated
  pessimistic: on all 87 gapped fills the live-achievable print was at or better than it.
- **Do not "improve" the stop order.** A price cap or passive working reopens the gap
  tail (measured: worst trade −572% capped / −253% laddered, vs −101% stop-market), and
  LULD-band pre-emption is 88% false alarms and loses in the recent regime.
- **Memory.** Ten years of minute bars does not fit one engine instance; shard per year.
- **Live data plumbing.** The `massive` NautilusTrader adapter covers live trades/quotes.
  There is no CenterPoint execution adapter; live execution needs a custom
  `ExecutionClient` or manual/DAS execution during the paper phase.

Live architecture, minimum components: (1) overnight scanner — prior-close map + real-time
trade stream, emits watchlist on 2x prints; (2) qualifier — cumulative premarket dollar
volume tracker, $1M gate, $1 as-traded floor; (3) locate manager — locate at
qualification, checkpoint increments, 2% fee gate, share ledger; (4) selector — top-5 by
premarket $vol at 09:30; (5) entry executor — 09:31 ladder with SSR handling; (6) stop
manager — 1.5x buy-stop on fill, stop-market with no price cap and no passive working,
market-on-reopen if triggered inside a halt, no re-entry; (7) exit executor — 15:57
ladder + 15:59 market fallback, half-day shift, halt-resumption contingency; (8) sizing
module — equity x 0.137 vs 0.005 x premarket volume, share trim at order time.

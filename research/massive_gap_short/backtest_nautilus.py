"""
NautilusTrader backtest: short 100%+ overnight gappers at the open, cover at the close.

Universe: events from candidates_overnight.parquet (overnight high >= 2x prior
close), restricted to a tradeable subset (see FILTERS below). Each event trades
a fixed $10k notional short via market orders on 1-minute RTH bars sourced from
Massive minute aggregates.

Entry: first RTH bar of the event day; the market order fills at that bar's
close tick (the last trade of the 09:30-09:31 minute), deliberately worse than
the official auction open since we react after the open.
Exit: first bar at/after 15:58 ET, filling at that bar's close tick (~15:58),
approximating a market-on-close without assuming the closing auction print.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import massive_api as m
from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model import AccountType
from nautilus_trader.model import Bar
from nautilus_trader.model import BarType
from nautilus_trader.model import Currency
from nautilus_trader.model import Equity
from nautilus_trader.model import InstrumentId
from nautilus_trader.model import Money
from nautilus_trader.model import OmsType
from nautilus_trader.model import OrderSide
from nautilus_trader.model import Price
from nautilus_trader.model import Quantity
from nautilus_trader.model import Symbol
from nautilus_trader.model import TimeInForce
from nautilus_trader.model import TraderId
from nautilus_trader.model import Venue
from nautilus_trader.trading import Strategy


DATA_DIR = Path(__file__).resolve().parent / "data"
ET = ZoneInfo("America/New_York")
UTC_TZ = UTC

VENUE = Venue("XNAS")
USD = Currency.from_str("USD")

NOTIONAL_PER_TRADE = 10_000.0
# NOTE: sub-cent per-share fees can't be represented as USD Money (2dp), so the
# engine runs commission-free and $0.005/share is deducted in bt_results.py.

# Tradeable-subset filters
MIN_OPEN_PRICE = 1.0
MIN_PM_DOLLAR_VOL = 1_000_000.0
MIN_RTH_BARS = 60
LATEST_FIRST_BAR = "10:30"
EARLIEST_LAST_BAR = "15:59"

ENTRY_CUTOFF_ET = (15, 0)
EXIT_TRIGGER_ET = (15, 58)


def load_events() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "candidates_overnight.parquet")
    ev = df[df["is_event"].fillna(False).astype(bool)].copy()
    # Timestamps are ET strings like "2026-08-21 09:30:00-04:00"; slice HH:MM
    ev["rth_first_t"] = ev["rth_first_ts"].str[11:16]
    ev["rth_last_t"] = ev["rth_last_ts"].str[11:16]
    ev = ev[
        (ev["open"] >= MIN_OPEN_PRICE)
        & (ev["pm_dollar_vol"] >= MIN_PM_DOLLAR_VOL)
        & (ev["n_rth_bars"] >= MIN_RTH_BARS)
        & (ev["rth_first_t"] <= LATEST_FIRST_BAR)
        & (ev["rth_last_t"] >= EARLIEST_LAST_BAR)
    ].copy()
    ev["qty"] = (NOTIONAL_PER_TRADE / ev["open"]).astype(int)
    ev = ev[ev["qty"] >= 1]  # drop split-adjusted price artifacts above the notional
    return ev.reset_index(drop=True)


def sane_symbol(ticker: str) -> str:
    return ticker.replace(".", "-").replace("/", "-")


def make_equity(symbol: str) -> Equity:
    sym = Symbol(symbol)
    return Equity(
        instrument_id=InstrumentId(sym, VENUE),
        raw_symbol=sym,
        currency=USD,
        price_precision=4,
        price_increment=Price.from_str("0.0001"),
        ts_event=0,
        ts_init=0,
        lot_size=Quantity.from_int(1),
    )


def build_bars(instrument: Equity, bar_type: BarType, ticker: str, prev_date: str, date: str) -> list[Bar]:
    data = m.minute_aggs(ticker, prev_date, date)  # served from disk cache
    results = data.get("results") or []
    day_930 = pd.Timestamp(f"{date} 09:30", tz="America/New_York").value // 1_000_000
    day_1600 = pd.Timestamp(f"{date} 16:00", tz="America/New_York").value // 1_000_000
    bars = []
    for r in results:
        t = r["t"]
        if not (day_930 <= t < day_1600):
            continue
        o, h, low, c = r["o"], r["h"], r["l"], r["c"]
        h = max(h, o, c)
        low = min(low, o, c)
        ts = t * 1_000_000 + 60_000_000_000  # bar close time, ns
        bars.append(
            Bar(
                bar_type=bar_type,
                open=instrument.make_price(round(o, 4)),
                high=instrument.make_price(round(h, 4)),
                low=instrument.make_price(round(low, 4)),
                close=instrument.make_price(round(c, 4)),
                volume=Quantity.from_int(max(int(r.get("v", 0)), 1)),
                ts_event=ts,
                ts_init=ts,
            ),
        )
    return bars


class GapShortConfig(StrategyConfig):
    def __new__(cls, *args, **kwargs):  # noqa: D102
        kwargs.pop("events", None)
        kwargs.pop("bar_types", None)
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, events: dict[tuple[str, str], int], bar_types: list[BarType]) -> None:
        super().__init__()
        self.events = events  # (instrument_id str, ET date str) -> qty
        self.bar_types = bar_types


class GapShort(Strategy):
    def __init__(self, config: GapShortConfig) -> None:
        super().__init__(config)
        self._events = config.events
        self._bar_types = config.bar_types
        self._entered: set[tuple[str, str]] = set()
        self._covered: set[tuple[str, str]] = set()

    def on_start(self) -> None:
        for bt in self._bar_types:
            self.subscribe_bars(bt)

    def on_bar(self, bar: Bar) -> None:
        iid = bar.bar_type.instrument_id
        dt_et = datetime.fromtimestamp(bar.ts_event / 1e9, tz=UTC_TZ).astimezone(ET)
        key = (str(iid), dt_et.strftime("%Y-%m-%d"))
        qty = self._events.get(key)
        if qty is None:
            return

        hm = (dt_et.hour, dt_et.minute)
        if key not in self._entered:
            if hm <= ENTRY_CUTOFF_ET:
                order = self.order_factory.market(
                    instrument_id=iid,
                    order_side=OrderSide.SELL,
                    quantity=Quantity.from_int(qty),
                    time_in_force=TimeInForce.GTC,
                )
                self.submit_order(order)
            self._entered.add(key)  # never retry entries
        elif key not in self._covered and hm >= EXIT_TRIGGER_ET:
            order = self.order_factory.market(
                instrument_id=iid,
                order_side=OrderSide.BUY,
                quantity=Quantity.from_int(qty),
                time_in_force=TimeInForce.GTC,
            )
            self.submit_order(order)
            self._covered.add(key)


def run_year(events: pd.DataFrame, year: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run one engine over a single calendar year of events (positions never span
    days, so sharding by year is exact and keeps memory bounded).
    """
    engine = BacktestEngine(
        config=BacktestEngineConfig(trader_id=TraderId.from_str("GAPSHORT-001")),
    )
    engine.add_venue(
        venue=VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money.from_str("10000000 USD")],
    )

    instruments: dict[str, Equity] = {}
    bar_types: dict[str, BarType] = {}
    all_bars: list[Bar] = []
    ev_map: dict[tuple[str, str], int] = {}

    for row in events.to_dict("records"):
        sym = sane_symbol(row["ticker"])
        if sym not in instruments:
            inst = make_equity(sym)
            instruments[sym] = inst
            engine.add_instrument(inst)
            bar_types[sym] = BarType.from_str(f"{inst.id}-1-MINUTE-LAST-EXTERNAL")
        inst = instruments[sym]
        bars = build_bars(inst, bar_types[sym], row["ticker"], row["prev_date"], row["date"])
        all_bars.extend(bars)
        ev_map[(str(inst.id), row["date"])] = int(row["qty"])

    all_bars.sort(key=lambda b: b.ts_init)
    print(f"[{year}] {len(events)} events, {len(instruments)} instruments, {len(all_bars)} bars")
    engine.add_data(all_bars)

    strategy = GapShort(GapShortConfig(events=ev_map, bar_types=list(bar_types.values())))
    engine.add_strategy(strategy)
    engine.run()

    fills = engine.generate_order_fills_report()
    positions = engine.generate_positions_report()
    engine.dispose()
    print(f"[{year}] fills={len(fills)} positions={len(positions)}")
    return fills.reset_index(), positions.reset_index()


def main() -> None:
    events = load_events()
    print(f"Backtesting {len(events)} events across {events['ticker'].nunique()} tickers")

    all_fills = []
    all_positions = []
    for year, ev_year in events.groupby(events["date"].str[:4]):
        fills, positions = run_year(ev_year, str(year))
        all_fills.append(fills)
        all_positions.append(positions)

    fills = pd.concat(all_fills, ignore_index=True)
    positions = pd.concat(all_positions, ignore_index=True)
    fills.to_csv(DATA_DIR / "bt_fills.csv", index=False)
    positions.to_csv(DATA_DIR / "bt_positions.csv", index=False)
    events.to_parquet(DATA_DIR / "bt_events.parquet", index=False)
    print(f"TOTAL fills={len(fills)} positions={len(positions)}")


if __name__ == "__main__":
    main()

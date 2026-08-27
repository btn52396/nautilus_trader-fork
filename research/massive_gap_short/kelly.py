"""
Empirical Kelly sizing for the gap short.

f = base_size / equity. Two estimates:
  per-trade Kelly: maximize E[log(1 + f * r_trade)]  (ignores same-day overlap)
  per-day Kelly:   maximize E[log(1 + f * d_day)]    (treats each event day as
                   one correlated bet: d = day PnL / $10k unit)

Plus a historical compounded replay at several fractions of the per-day Kelly,
including the historical max drawdown and per-year growth that sizing implies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
UNIT = 10_000.0


def kelly_opt(returns: np.ndarray) -> float:
    f_max = 0.999 / abs(returns.min())  # domain: 1 + f*r > 0
    grid = np.linspace(1e-4, f_max, 20_000)
    growth = [(np.log1p(f * returns)).mean() for f in grid]
    return float(grid[int(np.argmax(growth))])


def replay(daily_units: pd.Series, f: float) -> dict:
    """
    Compound $1 through the historical event days at fraction f (re-set daily).
    """
    rets = f * daily_units.to_numpy()
    equity = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return {
        "f": f,
        "terminal_multiple": float(equity[-1]),
        "max_dd": float(dd.min()),
        "worst_day": float(rets.min()),
        "log_growth_per_day": float(np.log1p(rets).mean()),
    }


def main() -> None:
    trades = pd.read_parquet(DATA_DIR / "bt_trades.parquet")
    r = trades["ret_on_notional"].to_numpy()

    daily = trades.groupby("date")["pnl"].sum() / UNIT  # day PnL in units of one trade's notional

    f_trade = kelly_opt(r)
    f_day = kelly_opt(daily.to_numpy())

    print(f"per-trade returns: n={len(r)} mean={r.mean():.4f} std={r.std():.4f} worst={r.min():.3f}")
    print(f"daily units:       n={len(daily)} mean={daily.mean():.4f} std={daily.std():.4f} worst={daily.min():.3f}")
    print(f"\nGaussian approx (mu/sigma^2), per-trade: {r.mean() / r.var():.3f}  <- misleading with this tail")
    print(f"Empirical full Kelly, per-trade bet:     {f_trade:.4f}")
    print(f"Empirical full Kelly, per-DAY bet:       {f_day:.4f}   <- the relevant one (overlapping positions)")

    print("\nHistorical compounded replay (fraction of equity as base_size, re-set daily):")
    rows = []
    for label, f in [
        ("full Kelly", f_day),
        ("1/2 Kelly", f_day / 2),
        ("1/4 Kelly", f_day / 4),
        ("1/8 Kelly", f_day / 8),
        ("$1.5k on $40k", 0.0375),
    ]:
        d = replay(daily, f)
        rows.append(
            {
                "sizing": label,
                "f (base/equity)": f"{f:.4f}",
                "$ at 40k": f"${f * 40_000:,.0f}",
                "terminal x (10y)": f"{d['terminal_multiple']:,.1f}x",
                "max DD": f"{d['max_dd']:.1%}",
                "worst day": f"{d['worst_day']:.1%}",
            },
        )
    print(pd.DataFrame(rows).to_string(index=False))

    # Per-year compounded growth at quarter Kelly for regime honesty
    f_q = f_day / 4
    yearly = (
        (1.0 + f_q * daily)
        .groupby(daily.index.str[:4])
        .prod()
        .sub(1.0)
    )
    print(f"\nPer-year compounded return at 1/4 Kelly (f={f_q:.4f}):")
    print(yearly.map(lambda x: f"{x:+.1%}").to_string())


if __name__ == "__main__":
    main()

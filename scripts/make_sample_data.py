"""Generate the synthetic sample price series used by the example.

Writes data/sample/sample_prices.csv with columns `date` and `close`.

These are simulated prices, not real market data. Daily log returns are drawn
from a normal distribution with a known volatility, using a fixed random seed,
so the file comes out identical every time. Because we know the true
volatility (20% a year), we can sanity-check the engine: the rolling
volatility it computes should hover around 0.20.

Run with:  uv run python scripts/make_sample_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

TRUE_ANNUAL_VOL = 0.20  # the volatility we simulate, so we know the "right answer"
TRADING_DAYS_PER_YEAR = 252
N_DAYS = 252  # one year of trading days
START_PRICE = 100.0
SEED = 42  # fixed seed -> reproducible file

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_prices.csv"


def make_sample_prices() -> pd.DataFrame:
    """Simulate one daily price path with a known volatility.

    Returns:
        DataFrame with columns `date` (business days, as YYYY-MM-DD strings)
        and `close` (price, rounded to 4 decimals).
    """
    rng = np.random.default_rng(SEED)

    # Convert annual volatility to daily: volatility scales with sqrt(time).
    daily_vol = TRUE_ANNUAL_VOL / np.sqrt(TRADING_DAYS_PER_YEAR)

    # One log return per day after the first. Zero mean (no drift) keeps it simple.
    log_returns = rng.normal(loc=0.0, scale=daily_vol, size=N_DAYS - 1)

    # Price_t = Price_0 * exp(sum of log returns up to t).
    # The leading 0.0 makes the first price exactly START_PRICE.
    cumulative_log_returns = np.concatenate([[0.0], np.cumsum(log_returns)])
    closes = START_PRICE * np.exp(cumulative_log_returns)

    # Business days (Mon-Fri); public holidays are ignored for simplicity.
    dates = pd.bdate_range(start="2025-01-02", periods=N_DAYS)

    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": closes.round(4)})


def main() -> None:
    """Generate the sample prices and write them to OUTPUT_PATH."""
    df = make_sample_prices()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

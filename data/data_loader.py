import pandas as pd
import yfinance as yf


def load_price_data(ticker: str, start: str) -> pd.Series:
    """
    Download historical adjusted close prices for a ticker.

    Args:
        ticker: Stock ticker symbol.
        start: Start date in YYYY-MM-DD format.

    Returns:
        A pandas Series of close prices indexed by date.
    """
    data = yf.download(ticker, start=start, auto_adjust=True)

    prices = data["Close"]
    if isinstance(prices, pd.DataFrame):
        prices = prices[ticker]

    return prices.dropna()


def estimate_gbm_parameters(prices: pd.Series) -> tuple[float, float]:
    """
    Estimate annualized drift and volatility from daily simple returns.

    Args:
        prices: Historical price series.

    Returns:
        Tuple of (mu, sigma), annualized drift and volatility.
    """
    returns = prices.pct_change().dropna()
    mu = returns.mean() * 252
    sigma = returns.std() * (252 ** 0.5)
    return mu, sigma

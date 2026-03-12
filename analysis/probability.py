import numpy as np
from scipy.stats import norm


def closed_form_probability(mu: float, sigma: float, T: float, threshold_multiple: float) -> float:
    """
    Compute the closed-form GBM probability:
    P(S_T >= threshold_multiple * S_0)

    Args:
        mu: Annualized drift.
        sigma: Annualized volatility.
        T: Time horizon in years.
        threshold_multiple: Target multiple of initial price, e.g. 1.2.

    Returns:
        Closed-form probability under GBM.
    """
    log_threshold = np.log(threshold_multiple)
    mean_log = (mu - 0.5 * sigma**2) * T
    std_log = sigma * np.sqrt(T)

    return norm.sf(log_threshold, loc=mean_log, scale=std_log)


def monte_carlo_probability(paths: np.ndarray, threshold_price: float) -> float:
    """
    Estimate probability from simulated terminal prices.

    Args:
        paths: Simulated price paths.
        threshold_price: Absolute price threshold.

    Returns:
        Monte Carlo estimated probability.
    """
    terminal_prices = paths[-1]
    return np.mean(terminal_prices >= threshold_price)


def monte_carlo_confidence_interval(prob: float, num_sim: int, z: float = 1.96) -> tuple[float, float]:
    """
    Compute approximate confidence interval for Monte Carlo Bernoulli estimate.

    Args:
        prob: Estimated Monte Carlo probability.
        num_sim: Number of simulations.
        z: Z critical value, default 1.96 for 95% CI.

    Returns:
        Lower and upper confidence interval bounds.
    """
    se = np.sqrt(prob * (1 - prob) / num_sim)
    lower = prob - z * se
    upper = prob + z * se
    return lower, upper

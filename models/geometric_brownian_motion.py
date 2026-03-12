import numpy as np


def simulate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    steps: int,
    num_sim: int,
    seed: int | None = 42
) -> np.ndarray:
    """
    Simulate GBM paths using Euler-exact discretization.

    Args:
        S0: Initial stock price.
        mu: Annualized drift.
        sigma: Annualized volatility.
        T: Time horizon in years.
        steps: Number of time steps.
        num_sim: Number of simulated paths.
        seed: Random seed for reproducibility.

    Returns:
        Matrix of simulated prices with shape (steps + 1, num_sim).
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / steps
    S = np.zeros((steps + 1, num_sim))
    S[0] = S0

    for t in range(1, steps + 1):
        shocks = np.random.normal(0, 1, num_sim)
        S[t] = S[t - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
        )

    return S

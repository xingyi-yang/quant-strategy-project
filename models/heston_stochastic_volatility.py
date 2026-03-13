import numpy as np


def simulate_heston_paths(
    S0: float,
    v0: float,
    mu: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    T: float,
    steps: int,
    num_sim: int,
    seed: int | None = 42
):
    """
    Simulate Heston stochastic volatility model paths.

    Args:
        S0: Initial stock price
        v0: Initial variance
        mu: Drift
        kappa: Mean reversion speed
        theta: Long-run variance
        xi: Volatility of volatility
        rho: Correlation between price and variance shocks
        T: Time horizon
        steps: Number of time steps
        num_sim: Number of simulations
        seed: Random seed

    Returns:
        S: simulated price paths
        v: simulated variance paths
    """

    if seed is not None:
        np.random.seed(seed)

    dt = T / steps

    S = np.zeros((steps + 1, num_sim))
    v = np.zeros((steps + 1, num_sim))

    S[0] = S0
    v[0] = v0

    for t in range(1, steps + 1):

        # generate correlated random variables
        Z1 = np.random.normal(0, 1, num_sim)    #stock price shock like "Market News" that hits the stock price directly.
        Z2 = np.random.normal(0, 1, num_sim)      # "Volatility-specific Noise"—random fluctuations in market uncertainty that have nothing to do with the direction of the stock price itself

        W1 = Z1    # assign it to stock price shock
        W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2   #Create the Correlated Variance shock

        # update variance
        v_prev = v[t - 1]
        v[t] = (
            v_prev
            + kappa * (theta - v_prev) * dt
            + xi * np.sqrt(np.maximum(v_prev, 0)) * np.sqrt(dt) * W2
        )

        # ensure variance stays positive
        v[t] = np.maximum(v[t], 0)

        # update stock price
        S[t] = S[t - 1] * np.exp(
            (mu - 0.5 * v_prev) * dt
            + np.sqrt(v_prev) * np.sqrt(dt) * W1
        )

    return S, v
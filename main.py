from data.data_loader import load_price_data, estimate_gbm_parameters
from models.geometric_brownian_motion import simulate_gbm_paths
from analysis.probability import (
    closed_form_probability,
    monte_carlo_probability,
    monte_carlo_confidence_interval,
)
from visualization.plots import plot_terminal_distribution, plot_sample_paths


def main() -> None:
    ticker = "AAPL"
    start = "2018-01-01"
    T = 1.0
    steps = 252
    num_sim = 10000
    threshold_multiple = 1.2

    # Load data and estimate parameters
    prices = load_price_data(ticker, start)
    mu, sigma = estimate_gbm_parameters(prices)

    # Initial price
    S0 = float(prices.iloc[-1])
    threshold_price = threshold_multiple * S0

    # Closed-form probability
    closed_prob = closed_form_probability(mu, sigma, T, threshold_multiple)

    # Monte Carlo simulation
    paths = simulate_gbm_paths(S0, mu, sigma, T, steps, num_sim)
    mc_prob = monte_carlo_probability(paths, threshold_price)
    ci_lower, ci_upper = monte_carlo_confidence_interval(mc_prob, num_sim)

    # Results
    print(f"Ticker: {ticker}")
    print(f"Initial price: {S0:.2f}")
    print(f"Annualized drift (mu): {mu:.4f}")
    print(f"Annualized volatility (sigma): {sigma:.4f}")
    print(f"Closed-form probability: {closed_prob:.4f}")
    print(f"Monte Carlo probability: {mc_prob:.4f}")
    print(f"95% CI for Monte Carlo estimate: ({ci_lower:.4f}, {ci_upper:.4f})")

    # Plots
    plot_sample_paths(paths, n_paths=20)
    plot_terminal_distribution(paths[-1], threshold_price)


if __name__ == "__main__":
    main()

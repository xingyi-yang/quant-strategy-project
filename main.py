from data.data_loader import load_price_data, estimate_gbm_parameters
from models.geometric_brownian_motion import simulate_gbm_paths
from models.heston_stochastic_volatility import simulate_heston_paths
from analysis.probability import (
    closed_form_probability,
    monte_carlo_probability,
    monte_carlo_confidence_interval,
)
from visualization.plots import plot_terminal_distribution, plot_sample_paths
import matplotlib.pyplot as plt

def main() -> None:
    ticker = "AAPL"
    start = "2018-01-01"
    T = 1.0
    steps = 252
    num_sim = 10000
    threshold_multiple = 1.2

    # Load data and estimate GBM parameters
    prices = load_price_data(ticker, start)
    mu, sigma = estimate_gbm_parameters(prices)

    # Initial price
    S0 = float(prices.iloc[-1])
    threshold_price = threshold_multiple * S0

    # ---------------- GBM ----------------
    # Closed-form probability
    closed_prob = closed_form_probability(mu, sigma, T, threshold_multiple)

    # Monte Carlo GBM simulation
    gbm_paths = simulate_gbm_paths(S0, mu, sigma, T, steps, num_sim)
    mc_prob_gbm = monte_carlo_probability(gbm_paths, threshold_price)
    ci_lower_gbm, ci_upper_gbm = monte_carlo_confidence_interval(mc_prob_gbm, num_sim)

    print("=== GBM Results ===")
    print(f"Ticker: {ticker}")
    print(f"Initial price: {S0:.2f}")
    print(f"Annualized drift (mu): {mu:.4f}")
    print(f"Annualized volatility (sigma): {sigma:.4f}")
    print(f"Closed-form probability: {closed_prob:.4f}")
    print(f"Monte Carlo probability: {mc_prob_gbm:.4f}")
    print(f"95% CI for Monte Carlo estimate: ({ci_lower_gbm:.4f}, {ci_upper_gbm:.4f})\n")

    # Plot GBM sample paths and terminal distribution
    plot_sample_paths(gbm_paths, n_paths=20, title="GBM Sample Paths")
    plot_terminal_distribution(gbm_paths[-1], threshold_price, title="GBM Terminal Distribution")

    # ---------------- Heston ----------------
    # Heston parameters
    v0 = sigma ** 2  # initial variance from GBM estimate
    kappa = 2.0      # mean reversion speed
    theta = sigma ** 2  # long-run variance
    xi = 0.3         # vol of vol
    rho = -0.7       # correlation

    # Monte Carlo Heston simulation
    heston_paths, heston_var = simulate_heston_paths(
        S0, v0, mu, kappa, theta, xi, rho, T, steps, num_sim
    )
    mc_prob_heston = monte_carlo_probability(heston_paths, threshold_price)
    ci_lower_heston, ci_upper_heston = monte_carlo_confidence_interval(mc_prob_heston, num_sim)

    print("=== Heston Results ===")
    print(f"Monte Carlo probability: {mc_prob_heston:.4f}")
    print(f"95% CI for Monte Carlo estimate: ({ci_lower_heston:.4f}, {ci_upper_heston:.4f})\n")

    # Plot Heston sample paths, variance paths, and terminal distribution
    plot_sample_paths(heston_paths, n_paths=20, title="Heston Stock Paths")
    plot_sample_paths(heston_var, n_paths=20, title="Heston Variance Paths", ylabel="Variance")
    plot_terminal_distribution(heston_paths[-1], threshold_price, title="Heston Terminal Distribution")

    # ---------------- Comparison Plots ----------------
    # Overlay GBM and Heston sample paths for visual comparison
    plt.figure(figsize=(10, 6))
    for i in range(min(10, gbm_paths.shape[1])):
        plt.plot(gbm_paths[:, i], color="blue", alpha=0.5, label="GBM" if i == 0 else "")
    for i in range(min(10, heston_paths.shape[1])):
        plt.plot(heston_paths[:, i], color="orange", alpha=0.5, label="Heston" if i == 0 else "")
    plt.title("Comparison: GBM vs Heston Sample Paths")
    plt.xlabel("Trading Days")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Overlay GBM and Heston terminal distributions
    plt.figure(figsize=(10, 6))
    plt.hist(gbm_paths[-1], bins=50, density=True, alpha=0.5, color="blue", label="GBM")
    plt.hist(heston_paths[-1], bins=50, density=True, alpha=0.5, color="orange", label="Heston")
    plt.axvline(threshold_price, color="red", linestyle="--", linewidth=2, label="Threshold")
    plt.title("Comparison: GBM vs Heston Terminal Distributions")
    plt.xlabel("Terminal Stock Price")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
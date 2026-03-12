import matplotlib.pyplot as plt
import numpy as np


def plot_terminal_distribution(terminal_prices: np.ndarray, threshold_price: float) -> None:
    """
    Plot histogram of simulated terminal prices.

    Args:
        terminal_prices: Simulated terminal stock prices.
        threshold_price: Price threshold to mark.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(terminal_prices, bins=50, density=True, alpha=0.7)
    plt.axvline(
        threshold_price,
        color="red",
        linestyle="--",
        linewidth=2,
        label="20% increase threshold"
    )
    plt.title("Distribution of Simulated Stock Prices After 1 Year")
    plt.xlabel("Terminal Stock Price")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_sample_paths(paths: np.ndarray, n_paths: int = 20) -> None:
    """
    Plot a subset of simulated GBM paths.

    Args:
        paths: Simulated price path matrix.
        n_paths: Number of sample paths to display.
    """
    plt.figure(figsize=(8, 5))
    for i in range(min(n_paths, paths.shape[1])):
        plt.plot(paths[:, i], linewidth=1)
    plt.title("Example Simulated GBM Price Paths")
    plt.xlabel("Trading Days")
    plt.ylabel("Stock Price")
    plt.tight_layout()
    plt.show()

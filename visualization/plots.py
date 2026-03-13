import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_terminal_distribution(
    terminal_prices: np.ndarray,
    threshold_price: float,
    title: str = "Distribution of Simulated Stock Prices After 1 Year",
    label_threshold: str = "Threshold"
) -> None:
    """
    Plot histogram of terminal stock prices with a threshold line.

    Args:
        terminal_prices: Simulated terminal stock prices.
        threshold_price: Price threshold to mark.
        title: Plot title.
        label_threshold: Label for the threshold line.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(terminal_prices, bins=50, density=True, alpha=0.7)
    plt.axvline(
        threshold_price,
        color="red",
        linestyle="--",
        linewidth=2,
        label=label_threshold
    )
    plt.title(title)
    plt.xlabel("Terminal Stock Price")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_sample_paths(
    paths: np.ndarray,
    n_paths: int = 20,
    title: str = "Example Simulated Price Paths",
    xlabel: str = "Trading Days",
    ylabel: str = "Stock Price"
) -> None:
    """
    Plot a subset of simulated price paths.

    Args:
        paths: Simulated price path matrix (steps+1 x num_sim).
        n_paths: Number of sample paths to display.
        title: Plot title.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
    """
    plt.figure(figsize=(8, 5))
    for i in range(min(n_paths, paths.shape[1])):
        plt.plot(paths[:, i], linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()
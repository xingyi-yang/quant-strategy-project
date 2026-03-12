import numpy  as np
import pandas as pd
import yfinance as yf
"""
Methodology: Estimating Probability of a 20% Stock Increase Using GBM

We model the stock price using a Geometric Brownian Motion (GBM) process,
which is widely used in quantitative finance because it ensures stock prices
remain positive and incorporates both expected return (drift) and volatility.

GBM formula:
    S_t = S_0 * exp((mu - 0.5 * sigma^2) * t + sigma * W_t)

Where:
    S_t   = stock price at time t
    S_0   = initial stock price
    mu    = drift (expected return)
    sigma = volatility
    W_t   = standard Brownian motion

Simulation Procedure:
1. Estimate drift (mu) and volatility (sigma) from historical returns.
2. Simulate many stock price paths over a 1-year horizon using the GBM model.
3. For each simulated path, check whether the final price satisfies:

       S_T >= 1.2 * S_0

4. Estimate the probability as the fraction of simulations where the
   stock price increases by at least 20%.

Result:
Based on the Monte Carlo simulation, the estimated probability that the
stock price increases by at least 20% within one year is approximately:

    P(S_T >= 1.2 * S_0) ≈ 0.551  (55.1%)

Interpretation:
Under the GBM assumption and the estimated parameters, there is about
a 55% probability that the stock price will rise by 20% or more within one year.
"""

data =  yf.download('AAPL', start = '2018-01-01')
prices = data["Close"]["AAPL"]


returns =  prices.pct_change().dropna()

mu = returns.mean() * 252   # drift
sigma = returns.std() * np.sqrt(252)   #volatility

# This takes the latest observed stock price. use this as starting pointof all simulating path
S0 = float(prices.iloc[-1])
#setup parameters
T = 1   # 1 year
steps = 252    # trading days
dt = T/steps   # time step
num_sim = 1000     #1000 possible future price paths

# creating a matrix, row is time, columns is simulation
S = np.zeros((steps+1, num_sim))
S[0] = S0

# Generate GBM paths
for t in range(1, steps+1):
    # generate 1000 independent shocks.
    Z = np.random.normal(0, 1, num_sim)
    S[t] = S[t-1] * np.exp((mu - (1/2)*sigma**2) * dt + sigma * np.sqrt(dt) * Z)

import matplotlib.pyplot as plt

plt.plot(S)
plt.title("GBM Simulated Price Paths")
plt.show()

#Extract the final simulated prices
final_prices = S[-1]

prob = np.mean(final_prices > 1.2*S0)

# Based on the GBM model, there is about a 55% probability that the stock will increase by 20% within one year.
print(prob)


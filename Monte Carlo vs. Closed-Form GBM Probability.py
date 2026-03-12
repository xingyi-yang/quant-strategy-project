"""
Quant Project Extension: Monte Carlo vs. Closed-Form GBM Probability

Goal:
Estimate the probability that a stock price will increase by at least 20%
within one year under a Geometric Brownian Motion (GBM) model.

Why this is stronger:
Instead of only using Monte Carlo simulation, we also compute the
closed-form probability implied by the GBM model's lognormal distribution.
Then we compare the two results and add a confidence interval for the
Monte Carlo estimate.

GBM Model:
    S_T = S_0 * exp((mu - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)

where:
    S_0   = initial stock price
    S_T   = stock price at time T
    mu    = drift
    sigma = volatility
    T     = time horizon in years
    Z     ~ N(0,1)

We want:
    P(S_T >= 1.2 * S_0)

--------------------------------------------------
1. Closed-Form Solution
--------------------------------------------------

Under GBM, log(S_T / S_0) is normally distributed:

    log(S_T / S_0) ~ N((mu - 0.5 * sigma^2) * T, sigma^2 * T)

So:

    P(S_T >= 1.2 * S_0)
    = P(log(S_T / S_0) >= log(1.2))

Standardizing gives:

    P(S_T >= 1.2 * S_0)
    = 1 - Phi(
        [log(1.2) - (mu - 0.5 * sigma^2) * T] / (sigma * sqrt(T))
      )

where Phi(.) is the standard normal CDF.

This gives an exact probability under the GBM assumption.

--------------------------------------------------
2. Monte Carlo Simulation
--------------------------------------------------

We also estimate the same probability by simulation:

1. Simulate many standard normal draws Z_i
2. Compute terminal prices S_T^(i) using the GBM formula
3. Count how many satisfy:

       S_T^(i) >= 1.2 * S_0

4. Estimate probability as:

       p_hat = (# of successes) / n_simulations

This provides a numerical approximation to the closed-form result.

--------------------------------------------------
3. Confidence Interval for Monte Carlo Estimate
--------------------------------------------------

Since each simulation is effectively a Bernoulli trial
(success = reached 20% gain, failure = did not), the Monte Carlo estimator
has approximate standard error:

    SE = sqrt(p_hat * (1 - p_hat) / n_simulations)

A 95% confidence interval is:

    p_hat ± 1.96 * SE

This tells us the simulation uncertainty due to having a finite number
of simulated paths.

--------------------------------------------------
4. Interpretation
--------------------------------------------------

- The closed-form result is the exact probability under GBM.
- The Monte Carlo result should be close to the closed-form result if the
  number of simulations is large.
- The confidence interval quantifies simulation noise.
- Any difference between the Monte Carlo and closed-form results should
  shrink as the number of simulations increases.

This makes the project stronger because it shows:
- knowledge of stochastic processes
- understanding of the lognormal distribution under GBM
- ability to validate simulation against theory
- awareness of statistical uncertainty in simulation results
"""

import numpy  as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

data =  yf.download('AAPL', start = '2018-01-01')
prices = data["Close"]["AAPL"]

returns =  prices.pct_change().dropna()
mu = returns.mean() * 252   # drift
sigma = returns.std() * np.sqrt(252)   #volatility

T = 1   # 1 year
steps = 252    # trading days
dt = T/steps   # time step

# -------------------------------
# 1. Closed-form GBM probability
# -------------------------------
log_threshold  = np.log(1.2)
mean_log = (mu- 1/2 * sigma**2) * T
std_log = sigma * np.sqrt(T)
closed_form_prob = norm.sf(log_threshold, loc = mean_log, scale = std_log)

# -------------------------------
# 2. Monte Carlo simulation
# -------------------------------

# This takes the latest observed stock price. use this as starting point of all simulating path
S0 = float(prices.iloc[-1])
#setup parameters
num_sim = 10000   #10000 possible future price paths

# creating a matrix, row is time, columns is simulation
S = np.zeros((steps+1, num_sim))
S[0] = S0

# Generate GBM paths
for t in range(1, steps+1):
    # generate 1000 independent shocks.
    Z = np.random.normal(0, 1, num_sim)
    S[t] = S[t-1] * np.exp((mu - (1/2)*sigma**2) * dt + sigma * np.sqrt(dt) * Z)

mc_prob = np.mean(S[-1] >= 1.2 * S0)
se = np.sqrt(mc_prob * (1-mc_prob)/num_sim)
ci_lower = mc_prob - 1.96 * se
ci_upper = mc_prob + 1.96 * se

# -------------------------------
# 4. Print results
# -------------------------------
print(f"Closed-form probability: {closed_form_prob:.4f}")
print(f"Monte Carlo probability: {mc_prob:.4f}")
print(f"95% CI for Monte Carlo estimate: ({ci_lower:.4f}, {ci_upper:.4f})")

"""
I modeled stock prices using Geometric Brownian Motion and estimated the
probability of a 20% gain over a one-year horizon. I solved the problem in
two ways: first by Monte Carlo simulation, and second by deriving the
closed-form probability from the lognormal distribution of GBM terminal prices.
I then compared the two results and added a confidence interval for the
simulation estimate to validate numerical accuracy.

The Monte Carlo estimate was very close to the closed-form probability,
which is expected under the GBM assumption when the number of simulations
is sufficiently large. This agreement helps validate that the simulation
was implemented correctly and that the numerical estimate is consistent
with the theoretical lognormal distribution implied by GBM.
"""


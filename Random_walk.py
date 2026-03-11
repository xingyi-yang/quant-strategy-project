import numpy as np
import matplotlib.pyplot as plt

def simulate_random_walks(n_steps, s_simulations, seed = 123):
    """
    Simulates and analyzes 1D symmetric random walks.
    Parameters:
    n_step (int): The number of steps in each individual walk.
    n_simulations (int): The number of independent walks to simulate for variance analysis.
    seed (int): Random seed for reproducibility.
    Returns: tuple: (single_walk_path, final_positions)
    """
    np.random.seed(seed)
    # --- PART 1: Single Walk Simulation ---
    # Generate 1000 steps: each step is either -1 or 1 with 50/50 probability
    steps = np.random.choice([-1, 1], size = n_steps)
    X = np.insert(np.cumsum(steps), 0, 0)

    # --- PART 2: Population Simulation (Vectorized) ---
    # Create a matrix where each row is a unique walk and each column is a step
    # This is much faster than a 'for' loop in Python
    all_walks = np.random.choice([-1, 1], size=(s_simulations, n_steps))
    # Sum across axis 1 (rows) to find the final position of each walker
    final_position = all_walks.sum(axis = 1)

    return X, steps, final_position

# --- Execution and Analysis ---
path, steps_data, finals = simulate_random_walks(1000, 1000)

# Visualization of the single walk trajectory
print(path.size)
plt.plot(path)
plt.title("Random Walk Path")
plt.xlabel("Step time")
plt.ylabel("Position")
plt.show()

#Statistical verification
print("Final position from one simulation:", path[-1])
print("Sample mean of steps:", np.mean(steps_data))
print("Sample variance of steps:", np.var(steps_data))

print("Variance of final position: ", np.var(finals))   # almost 1000

##########################simulate Brownian motion#################
# interval Wt2-Wt1 ~ N(0, t2-t1)
# Z ~  N(0, 1)  => sqrt(t2-t1)*Z ~ N(0, t2-t1)  => Wt2-Wt1 = sqrt(t2-t1)*Z
T = 1
n_steps = 1000
dt = T/n_steps

Z = np.random.normal(0, 1, n_steps)
dW = np.sqrt(dt) * Z     #Wt2-Wt1 = sqrt(t2-t1)*Z

W = np.insert(np.cumsum(dW), 0, 0)
time_grid = np.linspace(0, T, n_steps + 1)

plt.plot(time_grid, W)
plt.title("Brownian Motion")
plt.xlabel("Time")
plt.ylabel("W_t")
plt.show()

print("Approx final W_T:", W[-1])
print("Theoretical mean:", 0)
print("Theoretical variance for final Wt:", T)

########################Arithmetic Brownian motion##################
#The Stochastic Differential Equation (SDE)In your previous code,
# we had $dW_t$. Now, we define a new process $dX_t$:$$dX_t = \mu dt +
# \sigma dW_t$$$\mu dt$ (The Drift): This is the deterministic part.
# It represents the "trend" or the average return over time.$\sigma dW_t$
# (The Diffusion): This is the random part. It represents the "noise" or uncertainty (volatility).

# Parameters
T = 1
n_steps = 1000
dt = T/n_steps
mu = 0.5      # Expected annual return (Drift)
sigma = 0.2   # Volatility coefficient

# Generate Standard Brownian Motion (the "noise")
np.random.seed(42) # For reproducibility
Z = np.random.normal(0, 1, n_steps)
dW = np.sqrt(dt) * Z     #Wt2-Wt1 = sqrt(t2-t1)*Z

# Calculate the Change (dX) at each step
dX= mu*dt + sigma*dW

# Integrate (cumsum) to get the path
X = np.insert(np.cumsum(dX), 0, 0)
time_grid = np.linspace(0, T, n_steps + 1)


# 3. Calculate Theoretical Statistics
# E[Xt] = mu * t
expected_path = mu * time_grid
# StDev(Xt) = sigma * sqrt(t)
# We use 3 standard deviations for the "99.7% confidence interval"
upper_bound = expected_path + 3 * sigma * np.sqrt(time_grid)
lower_bound = expected_path - 3 * sigma * np.sqrt(time_grid)

# Plot the random realization
plt.plot(time_grid, X, label='Simulated Path ($X_t$)', color='#1f77b4', linewidth=1.5)

# Plot the Expected Mean
plt.plot(time_grid, expected_path, label='Expected Mean ($E[X_t]$)',
         color='red', linestyle='--')

# Plot the Variance/Uncertainty Fan
plt.fill_between(time_grid, lower_bound, upper_bound, color='gray', alpha=0.2,
                 label='3$\sigma$ Confidence Interval')

plt.title(f"Arithmetic Brownian Motion\n$\mu={mu}, \sigma={sigma}, T={T}$")
plt.xlabel("Time ($t$)")
plt.ylabel("Position ($X_t$)")
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.show()

# 5. Numerical Output
print(f"Final Position (Actual): {X[-1]:.4f}")
print(f"Theoretical Expectation: {mu * T:.4f}")
print(f"Theoretical Variance:    {(sigma**2) * T:.4f}")

########################Geometric Brownian motion##################
###simulating one stock-price path under Geometric Brownian Motion (GBM)
# and then plotting it against its theoretical mean and a lognormal “fan” of plausible values.

# 1. Setup Parameters
T = 1.0
n_steps = 1000
dt = T / n_steps
S0 = 100       # The stock starts at price 100.
mu = 0.1       # Annual Drift (10%)
sigma = 0.2    # Annual Volatility (20%)

# 2. Generate the Path (Vectorized using the Exact Solution)
np.random.seed(42)
time_grid = np.linspace(0, T, n_steps + 1)

# Standard Brownian Motion (W_t)
Z = np.random.normal(0, 1, n_steps)
dW = np.sqrt(dt) * Z
W = np.insert(np.cumsum(dW), 0, 0) #cumulative sum to build Wt

# Geometric Brownian Motion formula
# S_t = S0 * exp((mu - 0.5 * sigma^2) * t + sigma * W_t)
S = S0 * np.exp((mu - 0.5 * sigma**2) * time_grid + sigma * W)

# 3. Calculate Theoretical Statistics
# The Expected Value of GBM is E[St] = S0 * e^(mu * t)
expected_path = S0 * np.exp(mu * time_grid)

# For GBM, confidence intervals are calculated in "log-space"
# then exponentiated. This creates the characteristic skewed fan.
log_mean = (mu - 0.5 * sigma**2) * time_grid
log_sd = sigma * np.sqrt(time_grid)

# 3-sigma bounds (99.7% of paths in log-normal terms)
upper_bound = S0 * np.exp(log_mean + 3 * log_sd)
lower_bound = S0 * np.exp(log_mean - 3 * log_sd)

# 4. Visualization
plt.figure(figsize=(10, 6))

# Plot simulated path
plt.plot(time_grid, S, label='Simulated Stock Price ($S_t$)', color='#2ca02c', linewidth=1.5)

# Plot Expected Mean
plt.plot(time_grid, expected_path, label='Expected Mean ($E[S_t]$)',
         color='red', linestyle='--')

# Plot Confidence Interval
plt.fill_between(time_grid, lower_bound, upper_bound, color='gray', alpha=0.2,
                 label='3$\sigma$ Lognormal Fan')

plt.title(f"Geometric Brownian Motion\n$S_0={S0}, \mu={mu}, \sigma={sigma}$")
plt.xlabel("Time ($t$)")
plt.ylabel("Stock Price ($S$)")
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.show()

# 5. Numerical Output
print(f"Final Price: {S[-1]:.2f}")
print(f"Theoretical Expected Price: {S0 * np.exp(mu * T):.2f}")
print('')
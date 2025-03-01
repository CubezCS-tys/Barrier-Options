import numpy as np
import matplotlib.pyplot as plt

def up_and_out_put(S, K, T, r, sigma, B, num_steps=100):
    """
    Simulate a single path for an Up-and-Out Barrier Put Option.
    
    Parameters:
    - S: Initial stock price
    - K: Strike price
    - T: Time to maturity
    - r: Risk-free interest rate
    - sigma: Volatility
    - B: Barrier level
    - num_steps: Number of time steps in the simulation
    
    Returns:
    - Final payoff of the option (0 if the barrier is breached).
    - List of prices in the simulated path.
    - Boolean indicating if the barrier was breached.
    """
    dt = T / num_steps
    path = [S]
    barrier_breached = False

    # Simulate the path
    for _ in range(num_steps):
        Z = np.random.normal()
        next_price = path[-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
        path.append(next_price)

        # Check if the barrier is breached
        if next_price >= B:
            barrier_breached = True
            break  # Stop simulation as the option becomes worthless

    # Calculate payoff
    if barrier_breached:
        payoff = 0  # Option becomes worthless if the barrier is breached
    else:
        payoff = max(K - path[-1], 0)  # Put payoff if barrier is not breached

    return payoff, path, barrier_breached

# Parameters
S = 100   # Spot price
K = 100   # Strike price
T = 1     # Time to maturity in years
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility
B = 120    # Barrier level

# Run the simulation for one path
payoff, path, barrier_breached = up_and_out_put(S, K, T, r, sigma, B)

# Print results
print(f"Payoff: {payoff:.2f}")
print("Barrier breached:", barrier_breached)

# Plot the path
plt.figure(figsize=(10, 6))
plt.plot(path, label="Stock Price Path")
plt.axhline(y=B, color="red", linestyle="--", label="Barrier Level (B)")
plt.xlabel("Time Step")
plt.ylabel("Stock Price")
plt.title("Up-and-Out Barrier Put Option Simulation")
plt.legend()

# Show status on plot
if barrier_breached:
    plt.text(len(path) / 2, B + 5, "Barrier breached\nOption is worthless", color="red", fontsize=12, ha='center')
else:
    plt.text(len(path) - 1, path[-1], f"Payoff = {payoff:.2f}", color="green", fontsize=12, ha='right')

plt.show()

#If you think the stock will go down and below the barrier at least once and collect your profit as long as it doesnt go past the K price
import numpy as np
import matplotlib.pyplot as plt

def down_and_in_put(S, K, T, r, sigma, B, num_steps=100):
    """
    Simulate a single path for a Down-and-In Barrier Call Option.
    
    Parameters:
    - S: Initial stock price
    - K: Strike price
    - T: Time to maturity
    - r: Risk-free interest rate
    - sigma: Volatility
    - B: Barrier level
    - num_steps: Number of time steps in the simulation
    
    Returns:
    - Final payoff of the option (0 if the barrier is never breached).
    - List of prices in the simulated path.
    - Boolean indicating if the barrier was breached.
    """
    dt = T / num_steps
    path = [S]
    barrier_activated = False

    # Simulate the path
    for _ in range(num_steps):
        Z = np.random.normal()
        next_price = path[-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
        path.append(next_price)

        # Check if the barrier is breached
        if next_price <= B:
            barrier_activated = True  # Mark the option as active if the barrier is breached

    # Calculate payoff
    if barrier_activated:
        #payoff = max(path[-1] - K, 0)  # Call payoff if barrier was breached
        payoff = max(K - path[-1], 0)
    else:
        payoff = 0  # Payoff is zero if the barrier was never breached

    return payoff, path, barrier_activated

# Parameters
S = 100   # Spot price
K = 100   # Strike price
T = 1     # Time to maturity in years
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility
B = 90    # Barrier level

# Run the simulation for one path
payoff, path, barrier_activated = down_and_in_put(S, K, T, r, sigma, B)

# Print results
print(f"Payoff: {payoff:.2f}")
print("Barrier activated:", barrier_activated)

# Plot the path
plt.figure(figsize=(10, 6))
plt.plot(path, label="Stock Price Path")
plt.axhline(y=B, color="red", linestyle="--", label="Barrier Level (B)")
plt.xlabel("Time Step")
plt.ylabel("Stock Price")
plt.title("Down-and-In Barrier Option Simulation")
plt.legend()

# Show status on plot
if barrier_activated:
    plt.text(len(path) / 2, B+8, "Barrier activated\nOption is live", color="green", fontsize=12, ha='center')
    plt.text(len(path) - 1, path[-1], f"Payoff = {payoff:.2f}", color="green", fontsize=12, ha='right')
else:
    plt.text(len(path) / 2, B +1, "Barrier not reached\nOption has no value", color="red", fontsize=12, ha='center')

plt.show()

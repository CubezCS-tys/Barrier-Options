# If you think you will stay above your strike price to get your profit, barrier then becomes irrelevant
import numpy as np
import matplotlib.pyplot as plt

def down_and_out_call(S, K, T, r, sigma, B, num_steps=100):
    """
    Simulate a single path for a Down-and-Out Barrier Call Option.
    
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
        if next_price <= B:
            barrier_breached = True
            break  # Stop simulation as the option becomes worthless

    # Calculate payoff
    if barrier_breached:
        payoff = 0
    else:
        payoff = max(path[-1] - K, 0)  # Call payoff if barrier not breached

    return payoff, path, barrier_breached

# Parameters
S = 100   # Spot price
K = 100   # Strike price
T = 1     # Time to maturity in years
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility
B = 90    # Barrier level

# Run the simulation for one path
payoff, path, barrier_breached = down_and_out_call(S, K, T, r, sigma, B)

# Print results
print(f"Payoff: {payoff:.2f}")
print("Barrier breached:", barrier_breached)

# Plot the path
plt.figure(figsize=(10, 6))
plt.plot(path, label="Stock Price Path")
plt.axhline(y=B, color="red", linestyle="--", label="Barrier Level (B)")
plt.xlabel("Time Step")
plt.ylabel("Stock Price")
plt.title("Down-and-Out Barrier Option Simulation")
plt.legend()

# Show status on plot
if barrier_breached:
    plt.text(len(path) / 2, B+1, "Barrier breached\nOption is worthless", color="red", fontsize=12, ha='center')
else:
    plt.text(len(path) - 1, path[-1], f"Payoff = {payoff:.2f}", color="green", fontsize=12, ha='right')

plt.show()

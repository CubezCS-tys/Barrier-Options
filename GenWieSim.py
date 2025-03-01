import numpy as np
import time
import matplotlib.pyplot as plt  # Import pyplot for plotting

np.random.seed(42)

# Simulation parameters
a = 2
b = 4
T = 1.2
dt = 0.1

def simulate_generalised_wiener(a, b, T, dt):
    N = int(T / dt)  # Number of time steps
    times = np.linspace(0, T, N + 1)
    x = np.zeros(N + 1)  # Generalised Wiener process (with drift)
    w = np.zeros(N + 1)  # Standard Wiener process (without drift)
    drift = a * times    # Linear drift term

    # Enable interactive mode for dynamic plotting
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(1, N + 1):
        dz = np.random.normal(0, np.sqrt(dt))  # Increment from standard Wiener process
        dx = a * dt + b * dz  # Increment of the generalised process
        x[i] = x[i - 1] + dx  # Update generalised Wiener process
        w[i] = w[i - 1] + b*dz  # Update standard Wiener process

        # Clear the axis and plot updated data
        ax.clear()
        ax.plot(times[: i + 1], x[: i + 1], color="blue", label="Generalised Wiener Process")
        ax.plot(times[: i + 1], w[: i + 1], color="orange", label="Wiener Process (b dz)")
        ax.plot(times[: i + 1], drift[: i + 1], color="green", linestyle="--", label="Linear Drift (a dt)")
        ax.set_title("Generalised Wiener Process Simulation", fontsize=16)
        ax.set_xlabel("Time", fontsize=14)
        ax.set_ylabel("Value of x", fontsize=14)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        # Pause to allow the plot to update
        plt.pause(0.05)

    # Turn off interactive mode and show the final plot
    plt.ioff()
    plt.show()

# Run the simulation
simulate_generalised_wiener(a, b, T, dt)

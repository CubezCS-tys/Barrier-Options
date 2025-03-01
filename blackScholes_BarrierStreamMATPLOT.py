import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Preload default values for each option type
# Preload default values for each option type, using floats to avoid type mismatch
default_params = {
    "Up-and-Out Call": {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "B": 120.0, "num_steps": 100},
    "Down-and-Out Call": {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "B": 80.0, "num_steps": 100},
    "Up-and-In Call": {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "B": 120.0, "num_steps": 100},
    "Down-and-In Call": {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "B": 80.0, "num_steps": 100},
    "Up-and-Out Put": {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "B": 120.0, "num_steps": 100},
    "Down-and-Out Put": {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "B": 80.0, "num_steps": 100},
    "Up-and-In Put": {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "B": 120.0, "num_steps": 100},
    "Down-and-In Put": {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "B": 80.0, "num_steps": 100}
}

# Streamlit app settings
st.title("Barrier Option Simulator")

# Option selection
option_type = st.selectbox(
    "Select Barrier Option Type",
    list(default_params.keys())
)

# Preload parameters based on selected option
params = default_params[option_type]
S = st.sidebar.number_input("Initial Stock Price (S)", min_value=0.0, value=params["S"])
K = st.sidebar.number_input("Strike Price (K)", min_value=0.0, value=params["K"])
T = st.sidebar.number_input("Time to Maturity (T in years)", min_value=0.01, value=params["T"], step=0.01)
r = st.sidebar.number_input("Risk-Free Interest Rate (r)", min_value=0.0, value=params["r"], step=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.01, value=params["sigma"], step=0.01)
B = st.sidebar.number_input("Barrier Level (B)", min_value=0.0, value=params["B"])
num_steps = st.sidebar.number_input("Number of Time Steps", min_value=10, value=params["num_steps"])

# Button to run the simulation
if st.button("Run Simulation"):
    # Function to simulate the path and check for barrier options
    def simulate_barrier_option(S, K, T, r, sigma, B, option_type, num_steps=100):
        dt = T / num_steps
        path = [S]
        barrier_breached = False

        # Simulate path based on Geometric Brownian Motion
        for _ in range(num_steps):
            Z = np.random.normal()
            next_price = path[-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
            path.append(next_price)

            # Barrier conditions
            if option_type == "Up-and-Out Call" and next_price >= B:
                barrier_breached = True
                payoff = 0
                break
            elif option_type == "Down-and-Out Call" and next_price <= B:
                barrier_breached = True
                payoff = 0
                break
            elif option_type == "Up-and-In Call" and next_price >= B:
                barrier_breached = True
            elif option_type == "Down-and-In Call" and next_price <= B:
                barrier_breached = True
            elif option_type == "Up-and-Out Put" and next_price >= B:
                barrier_breached = True
                payoff = 0
                break
            elif option_type == "Down-and-Out Put" and next_price <= B:
                barrier_breached = True
                payoff = 0
                break
            elif option_type == "Up-and-In Put" and next_price >= B:
                barrier_breached = True
            elif option_type == "Down-and-In Put" and next_price <= B:
                barrier_breached = True

        # Payoff calculation for options that did not breach "Out" barriers or activated "In" barriers
        if not barrier_breached and "Out" in option_type:
            if "Call" in option_type:
                payoff = max(path[-1] - K, 0)
            elif "Put" in option_type:
                payoff = max(K - path[-1], 0)
        elif barrier_breached and "In" in option_type:
            if "Call" in option_type:
                payoff = max(path[-1] - K, 0)
            elif "Put" in option_type:
                payoff = max(K - path[-1], 0)
        elif not barrier_breached:
            payoff = 0

        return payoff, path, barrier_breached

    # Run simulation
    payoff, path, barrier_breached = simulate_barrier_option(S, K, T, r, sigma, B, option_type, num_steps)

    # Display results
    st.write(f"### Option Type: {option_type}")
    st.write(f"Barrier Level (B): {B}")
    st.write(f"Barrier Breached: {'Yes' if barrier_breached else 'No'}")
    st.write(f"Final Payoff: {payoff:.2f}")

    # Plot the simulated path
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(path, label="Stock Price Path")
    ax.axhline(y=B, color="red", linestyle="--", label="Barrier Level (B)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Stock Price")
    ax.set_title(f"{option_type} Simulation")
    ax.legend()

    # Show payoff status on the plot
    if barrier_breached and "Out" in option_type:
        ax.text(len(path) / 2, B + 5, "Barrier Breached\nOption is Worthless", color="red", fontsize=12, ha='center')
    elif not barrier_breached and "Out" in option_type:
        ax.text(len(path) / 2, B + 5, "Option is Live", color="green", fontsize=12, ha='center')
        ax.text(len(path) - 1, path[-1], f"Payoff = {payoff:.2f}", color="green", fontsize=12, ha='right')
    elif barrier_breached and "In" in option_type:
        ax.text(len(path) / 2, B + 5,"Barrier Activated\nOption is Live", color="green", fontsize=12, ha='center')
        ax.text(len(path) - 1, path[-1], f"Payoff = {payoff:.2f}", color="green", fontsize=12, ha='right')
    elif not barrier_breached and "In" in option_type:
         ax.text(len(path) / 2, B + 5, "Option is Worthless", color="red", fontsize=12, ha='center')


    # Display the plot in Streamlit
    st.pyplot(fig)


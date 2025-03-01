import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Preload default values for each option type
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
        barrier_activated = False

        # Simulate path based on Geometric Brownian Motion
        for _ in range(num_steps):
            Z = np.random.normal()
            next_price = path[-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
            path.append(next_price)

            # Barrier conditions
            if option_type == "Up-and-Out Call" and next_price >= B:
                return 0, path, True
            elif option_type == "Down-and-Out Call" and next_price <= B:
                return 0, path, True
            elif option_type == "Up-and-In Call" and next_price >= B:
                barrier_activated = True
            elif option_type == "Down-and-In Call" and next_price <= B:
                barrier_activated = True
            elif option_type == "Up-and-Out Put" and next_price >= B:
                return 0, path, True
            elif option_type == "Down-and-Out Put" and next_price <= B:
                return 0, path, True
            elif option_type == "Up-and-In Put" and next_price >= B:
                barrier_activated = True
            elif option_type == "Down-and-In Put" and next_price <= B:
                barrier_activated = True

        # Payoff calculation for options that did not breach "Out" barriers or activated "In" barriers
        if not barrier_activated and "Out" in option_type:
            payoff = max(path[-1] - K, 0) if "Call" in option_type else max(K - path[-1], 0)
        elif barrier_activated and "In" in option_type:
            payoff = max(path[-1] - K, 0) if "Call" in option_type else max(K - path[-1], 0)
        else:
            payoff = 0

        return payoff, path, barrier_activated

    # Run simulation
    payoff, path, barrier_breached = simulate_barrier_option(S, K, T, r, sigma, B, option_type, num_steps)

    # Display results
    st.write(f"### Option Type: {option_type}")
    st.write(f"Barrier Level (B): {B}")
    st.write(f"Barrier Breached: {'Yes' if barrier_breached else 'No'}")
    st.write(f"Final Payoff: {payoff:.2f}")

    # Plot the simulated path using Plotly for a sleeker, interactive plot
    fig = go.Figure()

    # Add stock price path
    fig.add_trace(go.Scatter(x=list(range(len(path))), y=path, mode="lines", name="Stock Price Path"))

    # Add barrier level as a horizontal line
    fig.add_trace(go.Scatter(x=[0, len(path) - 1], y=[B, B], mode="lines", name="Barrier Level (B)",
                             line=dict(color="red", dash="dash")))

    # Customize layout for a clean, sleek look
    fig.update_layout(
        title=f"{option_type} Simulation",
        xaxis_title="Time Step",
        yaxis_title="Stock Price",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="plotly_white"
    )

    # Show payoff status on the chart as an annotation
    if barrier_breached and "Out" in option_type:
        fig.add_annotation(x=len(path) / 2, y=B + 5, text="Barrier Breached - Option is Worthless", showarrow=False, font=dict(color="red", size=14))
    elif not barrier_breached and "Out" in option_type:
        fig.add_annotation(x=len(path) - 1, y=path[-1], text=f"Payoff = {payoff:.2f}", showarrow=True, arrowhead=2, font=dict(color="green", size=14))
    elif barrier_breached and "In" in option_type:
        fig.add_annotation(x=len(path) - 1, y=path[-1], text=f"Payoff = {payoff:.2f}", showarrow=True, arrowhead=2, font=dict(color="green", size=14))
    elif not barrier_breached and "In" in option_type:
        fig.add_annotation(x=len(path) / 2, y=B + 5, text="Barrier Not Breached - Option is Worthless", showarrow=False, font=dict(color="red", size=14))

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

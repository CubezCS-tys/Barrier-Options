import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Double Barrier Option Simulation", layout="centered")

# Title
st.title("Double Barrier Option Simulation")

# Sidebar for inputs
st.sidebar.header("Option Parameters")

# Option parameters
S0 = st.sidebar.number_input("Initial Stock Price (S₀)", value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0)
H_upper = st.sidebar.number_input("Upper Barrier (H_upper)", value=130.0, step=1.0)
H_lower = st.sidebar.number_input("Lower Barrier (H_lower)", value=70.0, step=1.0)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01)
r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05, step=0.01)
T = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0, step=0.1)
N = st.sidebar.number_input("Number of Time Steps (N)", value=252, step=1)
option_type = st.sidebar.selectbox("Option Type", ("Call", "Put"))
barrier_type = st.sidebar.selectbox("Barrier Type", ("Knock-Out", "Knock-In"))

# Run button
if st.button("Run Simulation"):
    # Validate inputs
    if H_lower >= H_upper:
        st.error("Lower barrier must be less than upper barrier.")
    else:
        # Time increment
        dt = T / N

        # Simulate a single price path
        np.random.seed()  # Seed based on system time
        Z = np.random.standard_normal(int(N))
        time = np.linspace(0, T, int(N)+1)
        S = np.zeros(int(N)+1)
        S[0] = S0

        # Generate the price path
        for t in range(1, int(N)+1):
            S[t] = S[t-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[t-1])

        # Check for barrier breach
        if barrier_type == "Knock-Out":
            breached = np.any((S >= H_upper) | (S <= H_lower))
            knocked_out = breached
            knocked_in = not breached
        elif barrier_type == "Knock-In":
            breached = np.any((S >= H_upper) | (S <= H_lower))
            knocked_in = breached
            knocked_out = not breached

        # Calculate payoff
        if option_type == "Call":
            intrinsic_value = max(S[-1] - K, 0)
        elif option_type == "Put":
            intrinsic_value = max(K - S[-1], 0)

        if barrier_type == "Knock-Out":
            if knocked_out:
                payoff = 0.0
                status = "Option Knocked Out"
            else:
                payoff = intrinsic_value * np.exp(-r * T)
                status = "Option Active"
        elif barrier_type == "Knock-In":
            if knocked_in:
                payoff = intrinsic_value * np.exp(-r * T)
                status = "Option Knocked In"
            else:
                payoff = 0.0
                status = "Option Not Knocked In"

        # Display results
        st.subheader("Simulation Results")
        st.write(f"**Option Status:** {status}")
        st.write(f"**Option Payoff:** {payoff:.2f}")

        # Plot the price path using Plotly
        fig = go.Figure()

        # Underlying asset price path
        fig.add_trace(go.Scatter(
            x=time,
            y=S,
            mode='lines',
            name='Underlying Asset Price'
        ))

        # Upper barrier
        fig.add_trace(go.Scatter(
            x=[time[0], time[-1]],
            y=[H_upper, H_upper],
            mode='lines',
            name='Upper Barrier',
            line=dict(color='red', dash='dash')
        ))

        # Lower barrier
        fig.add_trace(go.Scatter(
            x=[time[0], time[-1]],
            y=[H_lower, H_lower],
            mode='lines',
            name='Lower Barrier',
            line=dict(color='green', dash='dash')
        ))

        # Update layout
        fig.update_layout(
            title='Simulated Price Path with Double Barriers',
            xaxis_title='Time (Years)',
            yaxis_title='Stock Price',
            legend=dict(
                x=0,
                y=1,
                bgcolor='rgba(255,255,255,0)',
                bordercolor='rgba(0,0,0,0)'
            )
        )

        # Display the plot
        st.plotly_chart(fig)

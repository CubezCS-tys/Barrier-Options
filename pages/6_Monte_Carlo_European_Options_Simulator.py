# import streamlit as st
# import numpy as np
# import plotly.graph_objects as go
# import plotly.colors as colors
# from scipy.stats import norm

# # Streamlit app title
# st.set_page_config("Monte Carlo Sim", layout="wide")
# st.title("Monte Carlo Simulation for European Call Option")


# # Sidebar inputs for model parameters
# st.sidebar.header("Input Parameters")

# # Input parameters
# S0 = st.sidebar.number_input("Initial Stock Price (S0)", value=100.0, step=1.0)
# K = st.sidebar.number_input("Strike Price (K)", value=110.0, step=1.0)
# T = st.sidebar.number_input("Time to Maturity in Years (T)", value=1.0, step=0.1)
# r = st.sidebar.number_input("Risk-Free Interest Rate (r)", value=0.05, step=0.01)
# sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01)
# num_simulations = st.sidebar.number_input("Number of Simulations", value=1000, step=100)
# num_steps = st.sidebar.number_input("Number of Steps per Simulation", value=100, step=10)

# # Function to calculate Black-Scholes price for a European Call Option
# def black_scholes_call_price(S, K, T, r, sigma):
#     d1 = (np.log(S / K) + (r + 0.5 * sigma **2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
#     call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
#     return call_price

# # Run the Monte Carlo simulation when the user clicks the button
# if st.button("Run Simulation"):
#     dt = T / num_steps
#     #np.random.seed(0)
#     price_paths = np.zeros((num_steps + 1, num_simulations))
#     price_paths[0] = S0

#     for t in range(1, num_steps + 1):
#         Z = np.random.standard_normal(num_simulations)
#         price_paths[t] = price_paths[t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

#     # Calculate the payoff for each path at maturity
#     payoffs = np.maximum(price_paths[-1] - K, 0)
#     call_price = np.exp(-r * T) * np.mean(payoffs)

#     # Display the calculated option price
#     st.write(f"Monte Carlo Estimated Price for the European Call Option: ${call_price:.2f}")
    
#         # Analytical Black-Scholes price
#     bs_price = black_scholes_call_price(S0, K, T, r, sigma)
#     st.write(f"Analytical Black-Scholes Price: ${bs_price:.4f}")

#     # Plot the simulated paths
#     fig = go.Figure()
#     # Define a color palette for bolder lines
#     color_palette = colors.qualitative.Bold
#     for i in range(min(100, num_simulations)):  # Display up to 100 paths for clarity
#         fig.add_trace(go.Scatter(
#             x=np.linspace(0, T, num_steps + 1),
#             y=price_paths[:, i],
#             mode='lines',
#         line=dict(color=color_palette[i % len(color_palette)], width=1.5),
#         opacity=0.7  # Slightly increase opacity for clarity
#         ))

#     fig.update_layout(
#         title="Simulated Stock Price Paths",
#         xaxis_title="Time (Years)",
#         yaxis_title="Stock Price",
#         showlegend=False
#     )

#     # Display the plot in Streamlit
#     st.plotly_chart(fig)

#     # Histogram of payoffs
#     fig_payoff = go.Figure()
#     fig_payoff.add_trace(go.Histogram(x=payoffs, nbinsx=50, name="Payoff Distribution"))
#     fig_payoff.update_layout(
#         title="Distribution of Payoffs at Maturity",
#         xaxis_title="Payoff",
#         yaxis_title="Frequency",
#         showlegend=False
#     )

#     # Display payoff histogram
#     st.plotly_chart(fig_payoff)

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.colors as colors
from scipy.stats import norm

# Streamlit app title and configuration
#st.set_page_config(page_title="Monte Carlo Simulation", layout="wide")
st.set_page_config(page_title="Monte Carlo Simulation")
st.title("Monte Carlo Simulation for European Call Option")

# Sidebar inputs for model parameters
st.sidebar.header("Input Parameters")
st.sidebar.markdown("Adjust the parameters below to run the simulation.")

# Input parameters
S0 = st.sidebar.number_input("Initial Stock Price (S₀)", value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price (K)", value=110.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (Years)", value=1.0, step=0.1)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01)
num_simulations = st.sidebar.number_input("Number of Simulations", value=1000, step=100)
num_steps = st.sidebar.number_input("Number of Steps per Simulation", value=100, step=10)

# Function to calculate Black-Scholes price for a European Call Option
def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Expander for explaining the Monte Carlo formula
with st.expander("What is Monte Carlo Simulation?", expanded=True):
    st.subheader("Introduction to Monte Carlo Simulation")
    st.write("""
    Monte Carlo simulation is a numerical method used to estimate the value of financial instruments by simulating the underlying asset's price paths.
    The method is based on the stochastic differential equation (SDE):
    """)
    st.latex(r"""
    dS_t = \sigma S_t \, dt + \sigma S_t \, dW_t
    """)
    st.write("""
    **Where:**
    - $S_t$: The stock price at time $t$.
    - $\sigma$: The risk-free interest rate.
    - $\sigma$: The volatility of the stock price.
    - $dW_t$: Represents random noise (a Wiener process).

    This equation models the dynamics of a stock price influenced by a deterministic drift ($rS_t dt$) and random fluctuations ($\sigma S_t dW_t$).
    """)

    st.subheader("Discrete Formulation")
    st.write("""
    In discrete time, the stock price evolution can be approximated as:
    """)
    st.latex(r"""
    S_{t+1} = S_t \cdot \exp \left( (r - 0.5\sigma^2) \Delta t + \sigma \sqrt{\Delta t} \cdot Z \right)
    """)
    st.write("""
    **Where:**
    - $\Delta t$: Time step.
    - $Z$: A random variable sampled from a standard normal distribution.

    This formulation allows us to simulate multiple stock price paths over time.
    """)

    st.subheader("Steps in Monte Carlo Simulation")
    st.write("""
    1. **Simulate Stock Price Paths**:
        - Generate multiple potential stock price paths using the discrete SDE.
    2. **Calculate Payoffs at Maturity**:
        - For a European call option:
        """)
    st.latex(r"""
    \text{Payoff} = \max(S_T - K, 0)
    """)
    st.write("""
    3. **Discount Payoffs to Present Value**:
        - Calculate the option price by averaging the payoffs and discounting back:
        """)
    st.latex(r"""
    C = e^{-rT} \cdot \text{Mean(Payoffs)}
    """)
    st.write("""
    This approach provides an estimate of the option price, particularly useful when analytical solutions are not feasible.
    """)



# Run the Monte Carlo simulation when the user clicks the button
if st.button("Run Simulation"):
    dt = T / num_steps
    price_paths = np.zeros((num_steps + 1, num_simulations))
    price_paths[0] = S0

    # Simulate stock price paths
    for t in range(1, num_steps + 1):
        Z = np.random.standard_normal(num_simulations)
        price_paths[t] = price_paths[t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    # Calculate the payoff for each path at maturity
    payoffs = np.maximum(price_paths[-1] - K, 0)
    call_price = np.exp(-r * T) * np.mean(payoffs)

    # Display the calculated option price
    st.subheader("Results")
    st.write(f"**Monte Carlo Estimated Price:** ${call_price:.2f}")

    # Analytical Black-Scholes price
    bs_price = black_scholes_call_price(S0, K, T, r, sigma)
    st.write(f"**Analytical Black-Scholes Price:** ${bs_price:.4f}")

    # Plot the simulated paths
    st.subheader("Simulated Stock Price Paths")
    
    fig = go.Figure()
    color_palette = colors.qualitative.Bold
    for i in range(min(100, num_simulations)):  # Display up to 100 paths for clarity
        fig.add_trace(go.Scatter(
            x=np.linspace(0, T, num_steps + 1),
            y=price_paths[:, i],
            mode='lines',
            line=dict(color=color_palette[i % len(color_palette)], width=1.5),
            opacity=0.7
        ))

    fig.update_layout(
            title={
        "text": "Simulated Stock Price Paths",
        "x": 0.5,         # centres the title horizontally
        "xanchor": "center"
    },
        xaxis_title="Time (Years)",
        yaxis_title="Stock Price",
        showlegend=False
    )
    st.plotly_chart(fig)

    # Histogram of payoffs
    st.subheader("Payoff Distribution")
    fig_payoff = go.Figure()
    fig_payoff.add_trace(go.Histogram(x=payoffs, nbinsx=50, name="Payoff Distribution"))
    fig_payoff.update_layout(
        title="Distribution of Payoffs at Maturity",
        xaxis_title="Payoff",
        yaxis_title="Frequency",
        showlegend=False
    )
    st.plotly_chart(fig_payoff)

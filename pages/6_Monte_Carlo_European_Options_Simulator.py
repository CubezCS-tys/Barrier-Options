

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.colors as colors
from scipy.stats import norm

# Set page config for a wide layout
st.set_page_config("Monte Carlo Simulation", layout="wide")
st.title("Monte Carlo Simulation for European Options")

# ------------------------------
# Custom CSS for Info Boxes
# ------------------------------
st.markdown(
    """
    <style>
    .info-box {
        background-color: #f9f9f9;
        border-left: 5px solid #2c3e50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .info-box h4 {
        margin: 0;
        color: #2c3e50;
    }
    .info-box p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def create_info_box(title, value):
    """
    Returns HTML code for an info box with a title and value.
    """
    return f"""
    <div class="info-box">
        <h4>{title}</h4>
        <p>{value}</p>
    </div>
    """

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("Input Parameters")
S0 = st.sidebar.number_input("Initial Stock Price (S0)", value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price (K)", value=110.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (Years, T)", value=1.0, step=0.1)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01)
num_simulations = st.sidebar.number_input("Number of Simulations", value=1000, step=100)
num_steps = st.sidebar.number_input("Number of Steps per Simulation", value=100, step=10)
option_type = st.sidebar.selectbox("Option Type", ("Call", "Put"))

# ------------------------------
# Black-Scholes Formulas
# ------------------------------
def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# ------------------------------
# Run the Simulation
# ------------------------------
if st.button("Run Simulation"):
    dt = T / num_steps
    
    # Initialize array for simulated price paths
    price_paths = np.zeros((num_steps + 1, num_simulations))
    price_paths[0] = S0

    # Generate paths
    for t in range(1, num_steps + 1):
        Z = np.random.standard_normal(num_simulations)
        price_paths[t] = price_paths[t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )

    # Calculate payoffs at maturity
    if option_type == "Call":
        payoffs = np.maximum(price_paths[-1] - K, 0)
    else:
        payoffs = np.maximum(K - price_paths[-1], 0)

    # Monte Carlo estimated price
    mc_price = np.exp(-r * T) * np.mean(payoffs)
    std_error = np.std(payoffs) * np.exp(-r * T) / np.sqrt(num_simulations)
    ci_lower = mc_price - 1.96 * std_error
    ci_upper = mc_price + 1.96 * std_error

    # Analytical Black-Scholes price
    if option_type == "Call":
        bs_price = black_scholes_call_price(S0, K, T, r, sigma)
    else:
        bs_price = black_scholes_put_price(S0, K, T, r, sigma)

    # ------------------------------
    # Display Info Boxes in a Row (4 columns)
    # ------------------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(create_info_box("MC Price", f"${mc_price:,.2f}"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_info_box("95% CI", f"[${ci_lower:,.2f}, ${ci_upper:,.2f}]"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_info_box("BS Price", f"${bs_price:,.2f}"), unsafe_allow_html=True)
    with col4:
        st.markdown(create_info_box("Difference", f"${mc_price - bs_price:,.2f}"), unsafe_allow_html=True)

    # ------------------------------
    # Arrange Graphs in Two Rows of Two Columns
    # ------------------------------

    # Row 1: Simulated Paths and Payoff Histogram
    col1, col2 = st.columns(2)

    with col1:
        fig_paths = go.Figure()
        color_palette = colors.qualitative.Bold
        # Limit to 100 paths for clarity
        for i in range(min(num_simulations, 100)):
            fig_paths.add_trace(go.Scatter(
                x=np.linspace(0, T, num_steps + 1),
                y=price_paths[:, i],
                mode='lines',
                line=dict(color=color_palette[i % len(color_palette)], width=1.5),
                opacity=0.7
            ))
        fig_paths.update_layout(
            title="Simulated Stock Price Paths",
            xaxis_title="Time (Years)",
            yaxis_title="Stock Price",
            showlegend=False
        )
        st.plotly_chart(fig_paths, use_container_width=True)

    with col2:
        fig_payoff = go.Figure()
        fig_payoff.add_trace(go.Histogram(x=payoffs, nbinsx=50))
        fig_payoff.update_layout(
            title="Distribution of Payoffs at Maturity",
            xaxis_title="Payoff",
            yaxis_title="Frequency",
            showlegend=False
        )
        st.plotly_chart(fig_payoff, use_container_width=True)

    # Row 2: Final Stock Prices Distribution and Average Stock Price Path
    col3, col4 = st.columns(2)

    with col3:
        final_prices = price_paths[-1]
        fig_final = go.Figure()
        fig_final.add_trace(go.Histogram(x=final_prices, nbinsx=50))
        fig_final.update_layout(
            title="Distribution of Final Stock Prices",
            xaxis_title="Final Stock Price",
            yaxis_title="Frequency",
            showlegend=False
        )
        st.plotly_chart(fig_final, use_container_width=True)

    with col4:
        avg_path = np.mean(price_paths, axis=1)
        fig_avg = go.Figure()
        fig_avg.add_trace(go.Scatter(
            x=np.linspace(0, T, num_steps + 1),
            y=avg_path,
            mode='lines+markers',
            name='Average Path'
        ))
        fig_avg.update_layout(
            title="Average Stock Price Path Over Time",
            xaxis_title="Time (Years)",
            yaxis_title="Average Stock Price"
        )
        st.plotly_chart(fig_avg, use_container_width=True)

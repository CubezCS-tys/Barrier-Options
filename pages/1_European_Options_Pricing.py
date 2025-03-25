
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call_price, put_price, d1, d2

def greeks(S, K, T, r, sigma, d1, d2):
    call_delta = norm.cdf(d1)
    put_delta = norm.cdf(d1) - 1.0
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * norm.pdf(d1)
    call_theta = (
        -S * sigma * norm.pdf(d1) / (2.0 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2)
    )
    put_theta = (
        -S * sigma * norm.pdf(d1) / (2.0 * np.sqrt(T))
        + r * K * np.exp(-r * T) * norm.cdf(-d2)
    )
    call_rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    put_rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    return {
        "Call Delta": call_delta,
        "Put Delta": put_delta,
        "Gamma": gamma,
        "Vega": vega,
        "Call Theta": call_theta,
        "Put Theta": put_theta,
        "Call Rho": call_rho,
        "Put Rho": put_rho
    }

st.set_page_config(page_title="Black-Scholes Option Pricing Dashboard", layout="wide")
st.title("Black-Scholes Option Pricing Dashboard")

# -------------------------
# Sidebar for user inputs
# -------------------------
st.sidebar.header("Input Parameters")
S = st.sidebar.number_input("Spot Price (S)", min_value=0.0, value=50.0)
K = st.sidebar.number_input("Strike Price (K)", min_value=0.0, value=50.0)
T = st.sidebar.number_input("Time to Maturity (T, in years)", min_value=0.01, value=1.0, step=0.01)
r = st.sidebar.number_input("Risk-Free Interest Rate (r)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.3, step=0.01)

# Calculate option prices and Greeks at the given S
call_price, put_price, d1, d2 = black_scholes(S, K, T, r, sigma)
greeks_dict = greeks(S, K, T, r, sigma, d1, d2)

# -------------------------
# Top row: Call and Put values
# -------------------------
top_col1, top_col2 = st.columns(2)

with top_col1:
    # Call Value
    st.markdown(
        f"""
        <div style="
            background-color:#d4edda;
            border:2px solid #28a745;
            border-radius:15px;
            padding:20px;
            text-align:center;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        ">
        <h3 style="margin:0;">CALL Value</h3>
        <h1 style="margin:5px 0 0 0;">${call_price:.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

with top_col2:
    # Put Value
    st.markdown(
        f"""
        <div style="
            background-color:#f8d7da;
            border:2px solid #dc3545;
            border-radius:15px;
            padding:20px;
            text-align:center;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        ">
        <h3 style="margin:0;">PUT Value</h3>
        <h1 style="margin:5px 0 0 0;">${put_price:.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------
# Bottom row: Separate payoff + "today value" charts
# -------------------------
bottom_col1, bottom_col2 = st.columns(2)

# Make a range of underlying prices around the strike
S_range = np.linspace(0.5 * K, 1.5 * K, 100)

with bottom_col1:
    st.subheader("Call Option: Payoff vs. Value Today")

    # 1) Payoff at expiration
    call_payoff = np.maximum(S_range - K, 0)
    # 2) Black-Scholes value *today* for each S in S_range
    call_value_today = []
    for s_ in S_range:
        c_, _, _, _ = black_scholes(s_, K, T, r, sigma)
        call_value_today.append(c_)

    # Plot them together
    fig_call = go.Figure()
    fig_call.add_trace(go.Scatter(
        x=S_range, y=call_payoff,
        mode='lines', name='Payoff at Expiration',
        line=dict(color='green')
    ))
    fig_call.add_trace(go.Scatter(
        x=S_range, y=call_value_today,
        mode='lines', name='Option Value (Today)',
        line=dict(color='blue')
    ))
    fig_call.update_layout(
        xaxis_title="Underlying Price",
        yaxis_title="Value / Payoff",
        legend=dict(x=0, y=1)
    )
    st.plotly_chart(fig_call, use_container_width=True)



with bottom_col2:
    st.subheader("Put Option: Payoff vs. Value Today")

    # 1) Payoff at expiration
    put_payoff = np.maximum(K - S_range, 0)
    # 2) Black-Scholes value *today* for each S in S_range
    put_value_today = []
    for s_ in S_range:
        _, p_, _, _ = black_scholes(s_, K, T, r, sigma)
        put_value_today.append(p_)

    # Plot them together
    fig_put = go.Figure()
    fig_put.add_trace(go.Scatter(
        x=S_range, y=put_payoff,
        mode='lines', name='Payoff at Expiration',
        line=dict(color='red')
    ))
    fig_put.add_trace(go.Scatter(
        x=S_range, y=put_value_today,
        mode='lines', name='Option Value (Today)',
        line=dict(color='blue')
    ))
    fig_put.update_layout(
        xaxis_title="Underlying Price",
        yaxis_title="Value / Payoff",
        legend=dict(x=0, y=1)
    )
    st.plotly_chart(fig_put, use_container_width=True)



# -------------------------
# Greeks Table
# -------------------------
st.subheader("Option Greeks")
greeks_df = pd.DataFrame({
    "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
    "Call": [
        greeks_dict["Call Delta"],
        greeks_dict["Gamma"],
        greeks_dict["Vega"],
        greeks_dict["Call Theta"],
        greeks_dict["Call Rho"]
    ],
    "Put": [
        greeks_dict["Put Delta"],
        greeks_dict["Gamma"],
        greeks_dict["Vega"],
        greeks_dict["Put Theta"],
        greeks_dict["Put Rho"]
    ]
})
st.table(greeks_df)

# -------------------------
# Expandable Black-Scholes details
# -------------------------
with st.expander("See Black-Scholes Formula Details"):
    st.latex(r"""
    d_1 = \frac{\ln(S / K) + (r + 0.5 \sigma^2) T}{\sigma \sqrt{T}}
    """)
    st.latex(r"""
    d_2 = d_1 - \sigma \sqrt{T}
    """)
    st.latex(r"""
    \text{Call Price} = S\,N(d_1) - K\,e^{-rT}\,N(d_2)
    """)
    st.latex(r"""
    \text{Put Price} = K\,e^{-rT}\,N(-d_2) - S\,N(-d_1)
    """)


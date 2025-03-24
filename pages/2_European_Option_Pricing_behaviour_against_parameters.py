import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go

# ----------------------------
# Black-Scholes Formula Function
# ----------------------------
def black_scholes(S, K, T, r, sigma):
    """
    Black-Scholes formula for European Call and Put options.
    
    Parameters:
    S (float): Spot price of the underlying asset
    K (float): Strike price of the option
    T (float): Time to maturity (in years)
    r (float): Risk-free interest rate (as a decimal, e.g., 0.05 for 5%)
    sigma (float): Volatility of the underlying asset (as a decimal, e.g., 0.3 for 30%)
    
    Returns:
    tuple: (call_price, put_price)
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0, 0.0  # Handle invalid inputs gracefully
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return call_price, put_price

# ----------------------------
# Streamlit App Layout
# ----------------------------
st.set_page_config(page_title="Black-Scholes Option Pricing")

st.title("Black-Scholes Option Pricing behaviour against different parameters")

# ----------------------------
# Sidebar Inputs for Model Parameters
# ----------------------------
st.sidebar.header("Input Parameters")

# Input parameters with validation
S = st.sidebar.number_input("Spot Price (S)", min_value=0.01, value=50.0, step=0.1)
K = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=50.0, step=0.1)
T = st.sidebar.number_input("Time to Maturity (T, in years)", min_value=0.01, value=1.0, step=0.01)
r = st.sidebar.number_input("Risk-Free Interest Rate (r)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.3, step=0.01)

# ----------------------------
# Calculate Option Prices
# ----------------------------
call_price, put_price = black_scholes(S, K, T, r, sigma)

# ----------------------------
# Define a Function to Create Plotly Graphs
# ----------------------------
def create_plotly_graph(x, call_prices, put_prices, title, xlabel, ylabel, y_start_zero=False):
    """
    Creates a Plotly graph for Call and Put option prices.
    
    Parameters:
    x (array-like): X-axis data
    call_prices (array-like): Call option prices
    put_prices (array-like): Put option prices
    title (str): Title of the graph
    xlabel (str): Label for the X-axis
    ylabel (str): Label for the Y-axis
    y_start_zero (bool): If True, sets the Y-axis to start at zero
    
    Returns:
    plotly.graph_objects.Figure: The Plotly figure object
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=call_prices, mode='lines', name='Call Option Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x, y=put_prices, mode='lines', name='Put Option Price', line=dict(color='red')))
    
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        legend=dict(x=0.01, y=0.99),
        hovermode='x unified'
    )
    
    if y_start_zero:
        fig.update_yaxes(range=[0, max(max(call_prices), max(put_prices)) * 1.1])
    
    return fig

# ----------------------------
# Tabs for Each Plot
# ----------------------------
tabs = st.tabs(["Spot Price", "Time to Maturity", "Strike Price", "Volatility", "Interest Rate"])

# ----------------------------
# Tab 1: Option Price vs Spot Price
# ----------------------------
with tabs[0]:
    st.header("Option Price vs Spot Price")
    
    # User inputs for Spot Price range
    col1, col2 = st.columns(2)
    with col1:
        min_S = st.number_input("Minimum Spot Price (S)", min_value=0.01, value=0.01, step=0.1)
    with col2:
        max_S = st.number_input("Maximum Spot Price (S)", min_value=min_S, value=100.0, step=0.1)
    
    # Generate Spot Price data
    spot_prices_plot = np.linspace(min_S, max_S, 100)
    call_prices_spot = []
    put_prices_spot = []
    
    for S_value in spot_prices_plot:
        call_p, put_p = black_scholes(S_value, K, T, r, sigma)
        call_prices_spot.append(call_p)
        put_prices_spot.append(put_p)
    
    # Create Plotly graph
    fig_spot = create_plotly_graph(
        x=spot_prices_plot,
        call_prices=call_prices_spot,
        put_prices=put_prices_spot,
        title='Option Price vs Spot Price',
        xlabel='Spot Price (S)',
        ylabel='Option Price',
        y_start_zero=False
    )
    
    st.plotly_chart(fig_spot, use_container_width=True)

# ----------------------------
# Tab 2: Option Price vs Time to Maturity
# ----------------------------
with tabs[1]:
    st.header("Option Price vs Time to Maturity")
    
    # User inputs for Time to Maturity range
    col1, col2 = st.columns(2)
    with col1:
        min_T = st.number_input("Minimum Time to Maturity (T) [Years]", min_value=0.01, value=0.01, step=0.01)
    with col2:
        max_T = st.number_input("Maximum Time to Maturity (T) [Years]", min_value=min_T, value=1.8, step=0.01)
    
    # Generate Time to Maturity data
    time_to_maturity_plot = np.linspace(min_T, max_T, 100)
    call_prices_time = []
    put_prices_time = []
    
    for T_value in time_to_maturity_plot:
        call_p, put_p = black_scholes(S, K, T_value, r, sigma)
        call_prices_time.append(call_p)
        put_prices_time.append(put_p)
    
    # Create Plotly graph
    fig_time = create_plotly_graph(
        x=time_to_maturity_plot,
        call_prices=call_prices_time,
        put_prices=put_prices_time,
        title='Option Price vs Time to Maturity',
        xlabel='Time to Maturity (T) [Years]',
        ylabel='Option Price',
        y_start_zero=False
    )
    
    st.plotly_chart(fig_time, use_container_width=True)

# ----------------------------
# Tab 3: Option Price vs Strike Price
# ----------------------------
with tabs[2]:
    st.header("Option Price vs Strike Price")
    
    # User inputs for Strike Price range
    col1, col2 = st.columns(2)
    with col1:
        min_K = st.number_input("Minimum Strike Price (K)", min_value=0.01, value=0.01, step=0.1)
    with col2:
        max_K = st.number_input("Maximum Strike Price (K)", min_value=min_K, value=100.0, step=0.1)
    
    # Generate Strike Price data
    strike_prices_plot = np.linspace(min_K, max_K, 100)
    call_prices_strike = []
    put_prices_strike = []
    
    for K_value in strike_prices_plot:
        call_p, put_p = black_scholes(S, K_value, T, r, sigma)
        call_prices_strike.append(call_p)
        put_prices_strike.append(put_p)
    
    # Create Plotly graph
    fig_strike = create_plotly_graph(
        x=strike_prices_plot,
        call_prices=call_prices_strike,
        put_prices=put_prices_strike,
        title='Option Price vs Strike Price',
        xlabel='Strike Price (K)',
        ylabel='Option Price',
        y_start_zero=False
    )
    
    st.plotly_chart(fig_strike, use_container_width=True)

# ----------------------------
# Tab 4: Option Price vs Volatility
# ----------------------------
with tabs[3]:
    st.header("Option Price vs Volatility")
    
    # User inputs for Volatility range
    col1, col2 = st.columns(2)
    with col1:
        min_sigma = st.number_input("Minimum Volatility (σ) [%]", min_value=1.0, value=1.0, step=1.0)
    with col2:
        max_sigma = st.number_input("Maximum Volatility (σ) [%]", min_value=min_sigma, value=100.0, step=1.0)
    
    # Convert percentage to decimal
    min_sigma_dec = min_sigma / 100
    max_sigma_dec = max_sigma / 100
    
    # Generate Volatility data
    volatilities_plot = np.linspace(min_sigma_dec, max_sigma_dec, 100)
    call_prices_vol = []
    put_prices_vol = []
    
    for sigma_value in volatilities_plot:
        call_p, put_p = black_scholes(S, K, T, r, sigma_value)
        call_prices_vol.append(call_p)
        put_prices_vol.append(put_p)
    
    # Create Plotly graph
    fig_volatility = create_plotly_graph(
        x=volatilities_plot * 100,  # Convert to percentage
        call_prices=call_prices_vol,
        put_prices=put_prices_vol,
        title='Option Price vs Volatility',
        xlabel='Volatility (σ) [%]',
        ylabel='Option Price',
        y_start_zero=True  # Start Y-axis at zero
    )
    
    st.plotly_chart(fig_volatility, use_container_width=True)

# ----------------------------
# Tab 5: Option Price vs Risk-Free Interest Rate
# ----------------------------
with tabs[4]:
    st.header("Option Price vs Risk-Free Interest Rate")
    
    # User inputs for Risk-Free Interest Rate range
    col1, col2 = st.columns(2)
    with col1:
        min_r = st.number_input("Minimum Risk-Free Rate (r) [%]", min_value=0.0, value=0.0, step=0.5)
    with col2:
        max_r = st.number_input("Maximum Risk-Free Rate (r) [%]", min_value=min_r, value=20.0, step=0.5)
    
    # Convert percentage to decimal
    min_r_dec = min_r / 100
    max_r_dec = max_r / 100
    
    # Generate Risk-Free Interest Rate data
    interest_rates_plot = np.linspace(min_r_dec, max_r_dec, 100)
    call_prices_rate = []
    put_prices_rate = []
    
    for r_value in interest_rates_plot:
        call_p, put_p = black_scholes(S, K, T, r_value, sigma)
        call_prices_rate.append(call_p)
        put_prices_rate.append(put_p)
    
    # Create Plotly graph
    fig_interest = create_plotly_graph(
        x=interest_rates_plot * 100,  # Convert to percentage
        call_prices=call_prices_rate,
        put_prices=put_prices_rate,
        title='Option Price vs Risk-Free Interest Rate',
        xlabel='Risk-Free Interest Rate (r) [%]',
        ylabel='Option Price',
        y_start_zero=True  # Start Y-axis at zero
    )
    
    st.plotly_chart(fig_interest, use_container_width=True)


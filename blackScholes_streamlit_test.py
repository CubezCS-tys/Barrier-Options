'''
import numpy as np
from scipy.stats import norm
import streamlit as st
import plotly.graph_objects as go

# Black-Scholes formula for European Call and Put options
def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return call_price, put_price

# Streamlit App Title
st.title("Black-Scholes Option Pricing Calculator with 3D Visualization")

# Input fields for the model parameters
st.sidebar.header("Input Parameters")
K = st.sidebar.number_input("Strike Price (K)", min_value=0.0, value=100.0)
T = st.sidebar.number_input("Time to Maturity (T)", min_value=0.01, value=1.0, step=0.01)
r = st.sidebar.number_input("Risk-free Interest Rate (r)", min_value=0.0, value=0.05, step=0.01)

# Sliders for spot price and volatility range
spot_price_min = st.sidebar.number_input("Minimum Spot Price (S)", min_value=0.0, value=50.0)
spot_price_max = st.sidebar.number_input("Maximum Spot Price (S)", min_value=spot_price_min, value=150.0)
volatility_min = st.sidebar.number_input("Minimum Volatility (σ)", min_value=0.01, value=0.1, step=0.01)
volatility_max = st.sidebar.number_input("Maximum Volatility (σ)", min_value=volatility_min, value=0.5, step=0.01)

# Create ranges for spot price and volatility
S_values = np.linspace(spot_price_min, spot_price_max, 30)
sigma_values = np.linspace(volatility_min, volatility_max, 30)

# Create a meshgrid for 3D plotting
S_grid, sigma_grid = np.meshgrid(S_values, sigma_values)
call_price_grid = np.zeros_like(S_grid)
put_price_grid = np.zeros_like(S_grid)

# Calculate the option prices for each pair of (S, sigma)
for i in range(len(S_values)):
    for j in range(len(sigma_values)):
        call_price, put_price = black_scholes(S_grid[i, j], K, T, r, sigma_grid[i, j])
        call_price_grid[i, j] = call_price
        put_price_grid[i, j] = put_price

# Create 3D surface plot for Call Option Price
fig = go.Figure(data=[go.Surface(z=call_price_grid, x=S_values, y=sigma_values, colorscale='Viridis')])
fig.update_layout(title='Call Option Price Surface', scene = dict(
                    xaxis_title='Spot Price (S)',
                    yaxis_title='Volatility (σ)',
                    zaxis_title='Call Option Price'))

# Display the 3D plot in Streamlit
st.plotly_chart(fig)

# Create 3D surface plot for Put Option Price
fig2 = go.Figure(data=[go.Surface(z=put_price_grid, x=S_values, y=sigma_values, colorscale='Cividis')])
fig2.update_layout(title='Put Option Price Surface', scene = dict(
                    xaxis_title='Spot Price (S)',
                    yaxis_title='Volatility (σ)',
                    zaxis_title='Put Option Price'))

# Display the 3D plot in Streamlit
st.plotly_chart(fig2)
'''

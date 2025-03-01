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
st.set_page_config(page_title="Black-Scholes Option Pricing 3D plots")

st.title("Black-Scholes Option Pricing behaviour with varying volatility and spot price")

# Input fields for the model parameters
st.sidebar.header("Input Parameters")
K = st.sidebar.number_input("Strike Price (K)", min_value=0.0, value=100.0)
T = st.sidebar.number_input("Time to Maturity (T)", min_value=0.01, value=1.0, step=0.01)
r = st.sidebar.number_input("Risk-free Interest Rate (r)", min_value=0.0, value=0.05, step=0.01)

# Sliders for spot price and volatility range
spot_price_min = st.sidebar.number_input("Minimum Spot Price (S)", min_value=0.0, value=80.0)
spot_price_max = st.sidebar.number_input("Maximum Spot Price (S)", min_value=spot_price_min, value=120.0)
volatility_min = st.sidebar.number_input("Minimum Volatility (σ)", min_value=0.01, value=0.10, step=0.01)
volatility_max = st.sidebar.number_input("Maximum Volatility (σ)", min_value=volatility_min, value=0.30, step=0.01)

# Create concise ranges for spot price and volatility (fewer data points for a cleaner heatmap)
S_values = np.linspace(spot_price_min, spot_price_max, 10)
sigma_values = np.linspace(volatility_min, volatility_max, 10)

# Create a meshgrid for plotting
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
fig_call_surface = go.Figure(data=[go.Surface(z=call_price_grid, x=S_values, y=sigma_values, colorscale='Viridis')])
fig_call_surface.update_layout(
    title='Call Option Price Surface',
    width=700,  # Increase the width of the plot
    height=600,  # Increase the height of the plot
    scene=dict(
        xaxis_title='Spot Price (S)',
        yaxis_title='Volatility (σ)',
        zaxis_title='Call Option Price'
    ),
    margin=dict(l=0, r=0, t=50, b=0)  # Adjust margins for centering
)

# Display the 3D plot for Call Option Price in Streamlit
st.plotly_chart(fig_call_surface)

# Create 3D surface plot for Put Option Price
fig_put_surface = go.Figure(data=[go.Surface(z=put_price_grid, x=S_values, y=sigma_values, colorscale='Cividis')])
fig_put_surface.update_layout(
    title='Put Option Price Surface',
    width=700,  # Increase the width of the plot
    height=600,  # Increase the height of the plot
    scene=dict(
        xaxis_title='Spot Price (S)',
        yaxis_title='Volatility (σ)',
        zaxis_title='Put Option Price'
    ),
    margin=dict(l=0, r=0, t=50, b=0)  # Adjust margins for centering
)

# Display the 3D plot for Put Option Price in Streamlit
st.plotly_chart(fig_put_surface)


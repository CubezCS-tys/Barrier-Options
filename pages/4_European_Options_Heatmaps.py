

import numpy as np
from scipy.stats import norm
import streamlit as st
import matplotlib.pyplot as plt

# Black-Scholes formula for European Call and Put options
def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return call_price, put_price

st.set_page_config(page_title="Black-Scholes Option Pricing Using Heat maps", layout = "wide")
st.title("Heatmaps for Black-Scholes Option Pricing")

# Input fields for the model parameters
K = st.sidebar.number_input("Strike Price (K)", min_value=0.0, value=100.0)
T = st.sidebar.number_input("Time to Maturity (T)", min_value=0.01, value=1.0, step=0.01)
r = st.sidebar.number_input("Risk-free Interest Rate (r)", min_value=0.0, value=0.05, step=0.01)

spot_price_min = st.sidebar.number_input("Minimum Spot Price (S)", min_value=0.0, value=80.0)
spot_price_max = st.sidebar.number_input("Maximum Spot Price (S)", min_value=spot_price_min, value=120.0)
volatility_min = st.sidebar.number_input("Minimum Volatility (σ)", min_value=0.01, value=0.10, step=0.01)
volatility_max = st.sidebar.number_input("Maximum Volatility (σ)", min_value=volatility_min, value=0.30, step=0.01)

S_values = np.linspace(spot_price_min, spot_price_max, 10)
sigma_values = np.linspace(volatility_min, volatility_max, 10)

# Adjust meshgrid indexing
S_grid, sigma_grid = np.meshgrid(S_values, sigma_values, indexing='ij')

# Vectorize the option price calculations
call_price_grid, put_price_grid = black_scholes(S_grid, K, T, r, sigma_grid)

# Create Heatmap for Call Option Price
fig_call, ax_call = plt.subplots()
c_call = ax_call.pcolormesh(S_values, sigma_values, call_price_grid.T, shading='auto', cmap='viridis')
ax_call.set_title('Call Option Price Heatmap')
ax_call.set_xlabel('Spot Price (S)')
ax_call.set_ylabel('Volatility (σ)')
fig_call.colorbar(c_call, ax=ax_call, label='Call Option Price')

# Annotate the call heatmap
for i in range(len(S_values)):
    for j in range(len(sigma_values)):
        ax_call.text(S_values[i], sigma_values[j], f'{call_price_grid[i, j]:.2f}',
                     ha='center', va='center', color='white', fontsize=8)

# Create Heatmap for Put Option Price
fig_put, ax_put = plt.subplots()
c_put = ax_put.pcolormesh(S_values, sigma_values, put_price_grid.T, shading='auto', cmap='cividis')
ax_put.set_title('Put Option Price Heatmap')
ax_put.set_xlabel('Spot Price (S)')
ax_put.set_ylabel('Volatility (σ)')
fig_put.colorbar(c_put, ax=ax_put, label='Put Option Price')

# Annotate the put heatmap
for i in range(len(S_values)):
    for j in range(len(sigma_values)):
        ax_put.text(S_values[i], sigma_values[j], f'{put_price_grid[i, j]:.2f}',
                    ha='center', va='center', color='white', fontsize=8)

# Display the two heatmaps side by side using columns
col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig_call)
with col2:
    st.pyplot(fig_put)

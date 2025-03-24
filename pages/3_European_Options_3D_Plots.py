# import numpy as np
# from scipy.stats import norm
# import streamlit as st
# import plotly.graph_objects as go

# # Black-Scholes formula for European Call and Put options
# def black_scholes(S, K, T, r, sigma):
#     d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
    
#     call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
#     put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
#     return call_price, put_price

# # Streamlit App Title
# st.set_page_config(page_title="Black-Scholes Option Pricing 3D plots")

# st.title("Black-Scholes Option Pricing behaviour with varying volatility and spot price")

# # Input fields for the model parameters
# st.sidebar.header("Input Parameters")
# K = st.sidebar.number_input("Strike Price (K)", min_value=0.0, value=100.0)
# T = st.sidebar.number_input("Time to Maturity (T)", min_value=0.01, value=1.0, step=0.01)
# r = st.sidebar.number_input("Risk-free Interest Rate (r)", min_value=0.0, value=0.05, step=0.01)

# # Sliders for spot price and volatility range
# spot_price_min = st.sidebar.number_input("Minimum Spot Price (S)", min_value=0.0, value=80.0)
# spot_price_max = st.sidebar.number_input("Maximum Spot Price (S)", min_value=spot_price_min, value=120.0)
# volatility_min = st.sidebar.number_input("Minimum Volatility (σ)", min_value=0.01, value=0.10, step=0.01)
# volatility_max = st.sidebar.number_input("Maximum Volatility (σ)", min_value=volatility_min, value=0.30, step=0.01)

# # Create concise ranges for spot price and volatility (fewer data points for a cleaner heatmap)
# S_values = np.linspace(spot_price_min, spot_price_max, 10)
# sigma_values = np.linspace(volatility_min, volatility_max, 10)

# # Create a meshgrid for plotting
# S_grid, sigma_grid = np.meshgrid(S_values, sigma_values)
# call_price_grid = np.zeros_like(S_grid)
# put_price_grid = np.zeros_like(S_grid)

# # Calculate the option prices for each pair of (S, sigma)
# for i in range(len(S_values)):
#     for j in range(len(sigma_values)):
#         call_price, put_price = black_scholes(S_grid[i, j], K, T, r, sigma_grid[i, j])
#         call_price_grid[i, j] = call_price
#         put_price_grid[i, j] = put_price

# # Create 3D surface plot for Call Option Price
# fig_call_surface = go.Figure(data=[go.Surface(z=call_price_grid, x=S_values, y=sigma_values, colorscale='Viridis')])
# fig_call_surface.update_layout(
#     title='Call Option Price Surface',
#     width=700,  # Increase the width of the plot
#     height=600,  # Increase the height of the plot
#     scene=dict(
#         xaxis_title='Spot Price (S)',
#         yaxis_title='Volatility (σ)',
#         zaxis_title='Call Option Price'
#     ),
#     margin=dict(l=0, r=0, t=50, b=0)  # Adjust margins for centering
# )

# # Display the 3D plot for Call Option Price in Streamlit
# st.plotly_chart(fig_call_surface)

# # Create 3D surface plot for Put Option Price
# fig_put_surface = go.Figure(data=[go.Surface(z=put_price_grid, x=S_values, y=sigma_values, colorscale='Cividis')])
# fig_put_surface.update_layout(
#     title='Put Option Price Surface',
#     width=700,  # Increase the width of the plot
#     height=600,  # Increase the height of the plot
#     scene=dict(
#         xaxis_title='Spot Price (S)',
#         yaxis_title='Volatility (σ)',
#         zaxis_title='Put Option Price'
#     ),
#     margin=dict(l=0, r=0, t=50, b=0)  # Adjust margins for centering
# )

# # Display the 3D plot for Put Option Price in Streamlit
# st.plotly_chart(fig_put_surface)

import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go

# ----------------------------
# Black-Scholes Formula Function
# ----------------------------
def black_scholes(S, K, T, r, sigma):
    """Returns (call_price, put_price) using the Black-Scholes formula."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0, 0.0  # Handle invalid inputs gracefully
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price  = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call_price, put_price

# ----------------------------
# Create Plotly line-plot function (for 2D plots)
# ----------------------------
def create_plotly_graph(x, call_prices, put_prices, title, xlabel, ylabel, y_start_zero=False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=call_prices, mode='lines', 
        name='Call Option Price', line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=x, y=put_prices, mode='lines', 
        name='Put Option Price', line=dict(color='red')
    ))
    
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
# Create Plotly surface-plot function (for 3D plots)
# ----------------------------
def create_3d_surface(X, Y, Z, x_label, y_label, z_label, title):
    """
    X, Y, Z must be 2D arrays (meshgrid) of the same shape.
    We'll plot Z as a function of (X, Y).
    """
    fig = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',  # or 'Plasma', etc.
            opacity=0.9
        )
    ])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig

# ----------------------------
# Streamlit App Layout
# ----------------------------
st.set_page_config(page_title="Black-Scholes Option Pricing", layout = "wide")
st.title("Black-Scholes Option Pricing behaviour against different parameters")

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("Input Parameters")

S    = st.sidebar.number_input("Spot Price (S)", min_value=0.01, value=50.0, step=0.1)
K    = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=50.0, step=0.1)
T    = st.sidebar.number_input("Time to Maturity (T, in years)", min_value=0.01, value=1.0, step=0.01)
r    = st.sidebar.number_input("Risk-Free Interest Rate (r)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
sigma= st.sidebar.number_input("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.3, step=0.01)

call_price, put_price = black_scholes(S, K, T, r, sigma)

# ----------------------------
# 3D Surface Plots Section
# ----------------------------
st.header("3D Surface Plots")
st.write("Select a parameter to vary on the Y-axis (alongside Spot Price on the X-axis).")

param_3d = st.selectbox(
    "Choose the second dimension:",
    ("Strike Price", "Time to Maturity", "Volatility", "Interest Rate")
)

# Define S range
st.subheader("Spot Price Range")
col1, col2 = st.columns(2)
with col1:
    min_S_3d = st.number_input("Min S (for 3D)", min_value=0.01, value=10.0, step=1.0)
with col2:
    max_S_3d = st.number_input("Max S (for 3D)", min_value=min_S_3d, value=100.0, step=1.0)
S_points = 30  # resolution
S_grid   = np.linspace(min_S_3d, max_S_3d, S_points)

# Depending on the param_3d, define the range for that parameter
param_points = 30

if param_3d == "Strike Price":
    st.subheader("Strike Price Range")
    col3, col4 = st.columns(2)
    with col3:
        min_K_3d = st.number_input("Min K (for 3D)", min_value=0.01, value=10.0, step=1.0)
    with col4:
        max_K_3d = st.number_input("Max K (for 3D)", min_value=min_K_3d, value=100.0, step=1.0)
    param_grid = np.linspace(min_K_3d, max_K_3d, param_points)

    # Build 2D mesh
    X, Y = np.meshgrid(S_grid, param_grid)  # X ~ S, Y ~ K
    Z_call = np.zeros_like(X)
    Z_put  = np.zeros_like(X)

    # Fill Z_call, Z_put
    for i in range(param_points):
        for j in range(S_points):
            c, p = black_scholes(X[i,j], Y[i,j], T, r, sigma)
            Z_call[i,j] = c
            Z_put[i,j]  = p

    # Plot side by side
    col_call, col_put = st.columns(2)

    with col_call:
        fig_call_3d = create_3d_surface(
            X, Y, Z_call,
            x_label="Spot Price (S)",
            y_label="Strike Price (K)",
            z_label="Call Price",
            title="Call Option Price Surface"
        )
        st.plotly_chart(fig_call_3d, use_container_width=True)

    with col_put:
        fig_put_3d = create_3d_surface(
            X, Y, Z_put,
            x_label="Spot Price (S)",
            y_label="Strike Price (K)",
            z_label="Put Price",
            title="Put Option Price Surface"
        )
        st.plotly_chart(fig_put_3d, use_container_width=True)

elif param_3d == "Time to Maturity":
    st.subheader("Time to Maturity Range (Years)")
    col3, col4 = st.columns(2)
    with col3:
        min_T_3d = st.number_input("Min T (for 3D)", min_value=0.01, value=0.1, step=0.1)
    with col4:
        max_T_3d = st.number_input("Max T (for 3D)", min_value=min_T_3d, value=2.0, step=0.1)
    param_grid = np.linspace(min_T_3d, max_T_3d, param_points)

    X, Y = np.meshgrid(S_grid, param_grid)  # X ~ S, Y ~ T
    Z_call = np.zeros_like(X)
    Z_put  = np.zeros_like(X)

    for i in range(param_points):
        for j in range(S_points):
            c, p = black_scholes(X[i,j], K, Y[i,j], r, sigma)
            Z_call[i,j] = c
            Z_put[i,j]  = p

    col_call, col_put = st.columns(2)

    with col_call:
        fig_call_3d = create_3d_surface(
            X, Y, Z_call,
            x_label="Spot Price (S)",
            y_label="Time to Maturity (T)",
            z_label="Call Price",
            title="Call Option Price Surface"
        )
        st.plotly_chart(fig_call_3d, use_container_width=True)

    with col_put:
        fig_put_3d = create_3d_surface(
            X, Y, Z_put,
            x_label="Spot Price (S)",
            y_label="Time to Maturity (T)",
            z_label="Put Price",
            title="Put Option Price Surface"
        )
        st.plotly_chart(fig_put_3d, use_container_width=True)

elif param_3d == "Volatility":
    st.subheader("Volatility Range (%)")
    col3, col4 = st.columns(2)
    with col3:
        min_sigma_pct_3d = st.number_input("Min σ (for 3D) [%]", min_value=1.0, value=5.0, step=1.0)
    with col4:
        max_sigma_pct_3d = st.number_input("Max σ (for 3D) [%]", min_value=min_sigma_pct_3d, value=100.0, step=1.0)
    
    min_sigma_dec_3d = min_sigma_pct_3d / 100
    max_sigma_dec_3d = max_sigma_pct_3d / 100
    param_grid = np.linspace(min_sigma_dec_3d, max_sigma_dec_3d, param_points)

    X, Y = np.meshgrid(S_grid, param_grid)  # X ~ S, Y ~ sigma
    Z_call = np.zeros_like(X)
    Z_put  = np.zeros_like(X)

    for i in range(param_points):
        for j in range(S_points):
            c, p = black_scholes(X[i,j], K, T, r, Y[i,j])
            Z_call[i,j] = c
            Z_put[i,j]  = p

    col_call, col_put = st.columns(2)

    with col_call:
        fig_call_3d = create_3d_surface(
            X, Y, Z_call,
            x_label="Spot Price (S)",
            y_label="Volatility (σ)",
            z_label="Call Price",
            title="Call Option Price Surface"
        )
        st.plotly_chart(fig_call_3d, use_container_width=True)

    with col_put:
        fig_put_3d = create_3d_surface(
            X, Y, Z_put,
            x_label="Spot Price (S)",
            y_label="Volatility (σ)",
            z_label="Put Price",
            title="Put Option Price Surface"
        )
        st.plotly_chart(fig_put_3d, use_container_width=True)

else:  # param_3d == "Interest Rate"
    st.subheader("Risk-Free Rate Range (%)")
    col3, col4 = st.columns(2)
    with col3:
        min_r_pct_3d = st.number_input("Min r (for 3D) [%]", min_value=0.0, value=0.0, step=0.5)
    with col4:
        max_r_pct_3d = st.number_input("Max r (for 3D) [%]", min_value=min_r_pct_3d, value=20.0, step=0.5)
    
    min_r_dec_3d = min_r_pct_3d / 100
    max_r_dec_3d = max_r_pct_3d / 100
    param_grid = np.linspace(min_r_dec_3d, max_r_dec_3d, param_points)

    X, Y = np.meshgrid(S_grid, param_grid)  # X ~ S, Y ~ r
    Z_call = np.zeros_like(X)
    Z_put  = np.zeros_like(X)

    for i in range(param_points):
        for j in range(S_points):
            c, p = black_scholes(X[i,j], K, T, Y[i,j], sigma)
            Z_call[i,j] = c
            Z_put[i,j]  = p

    col_call, col_put = st.columns(2)

    with col_call:
        fig_call_3d = create_3d_surface(
            X, Y, Z_call,
            x_label="Spot Price (S)",
            y_label="Risk-Free Rate (r)",
            z_label="Call Price",
            title="Call Option Price Surface"
        )
        st.plotly_chart(fig_call_3d, use_container_width=True)

    with col_put:
        fig_put_3d = create_3d_surface(
            X, Y, Z_put,
            x_label="Spot Price (S)",
            y_label="Risk-Free Rate (r)",
            z_label="Put Price",
            title="Put Option Price Surface"
        )
        st.plotly_chart(fig_put_3d, use_container_width=True)

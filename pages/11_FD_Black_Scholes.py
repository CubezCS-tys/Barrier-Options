#MAIN
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import interp1d
from scipy.linalg import lu
import time

# Black-Scholes formula for analytical solution
def black_scholes(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) + 1e-12)
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "Call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "Put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# Forward Euler finite difference method
def forward_euler(S0, K, T, r, sigma, dS, dt, option_type):
    S_max = 2*max(S0,K)*np.exp(r*T)
    M = int(S_max / dS)
    N = int(T / dt)
    dS = S_max / M
    dt = T / N


    matval = np.zeros((M + 1, N + 1))
    vetS = np.linspace(0, S_max, M + 1)

    # Boundary conditions
    if option_type == "Call":
        matval[:, -1] = np.maximum(vetS - K, 0)
        matval[0, :] = 0
        matval[-1, :] = S_max - K * np.exp(-r * (T - np.linspace(0, T, N + 1)))
    elif option_type == "Put":
        matval[:, -1] = np.maximum(K - vetS, 0)
        matval[0, :] = K * np.exp(-r * (T - np.linspace(0, T, N + 1)))
        matval[-1, :] = 0

    # Coefficients
    a = 0.5 * dt * (sigma**2 * np.arange(M + 1) - r) * np.arange(M + 1)
    b = 1 - dt * (sigma**2 * np.arange(M + 1)**2 + r)
    c = 0.5 * dt * (sigma**2 * np.arange(M + 1) + r) * np.arange(M + 1)

    # Time-stepping
    for j in range(N, 0, -1):
        for i in range(1, M):
            matval[i, j - 1] = (
                a[i] * matval[i - 1, j]
                + b[i] * matval[i, j]
                + c[i] * matval[i + 1, j]
            )

    price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
    price = price_interp(S0)

        
    return price, vetS, matval[:, 0]



def backward_euler(S0, K, r, T, sigma, dS, dt, option_type):
    # set up grid and adjust increments if necessary
    Smax = 2*max(S0,K)*np.exp(r*T)
    M = round(Smax / dS)
    dS = Smax / M
    N = round(T / dt)
    dt = T / N
    matval = np.zeros((M + 1, N + 1))
    vetS = np.linspace(0, Smax, M + 1)
    veti = np.arange(0, M + 1)
    vetj = np.arange(0, N + 1)
    
    # Boundary conditions
    if option_type == "Call":
        matval[:, -1] = np.maximum(vetS - K, 0)
        matval[0, :] = 0
        #matval[-1, :] = S_max - K * np.exp(-r * (T - np.linspace(0, T, N + 1)))
        matval[-1, :] = Smax - K * np.exp(-r * dt * (N - vetj))
    elif option_type == "Put":
        matval[:, -1] = np.maximum(K - vetS, 0)
        matval[0, :] = K * np.exp(-r * (T - np.linspace(0, T, N + 1)))
        matval[-1, :] = 0
    
    # set up the tridiagonal coefficients matrix
    a = 0.5 * (r * dt * veti - sigma**2 * dt * (veti**2))
    b = 1 + sigma**2 * dt * (veti**2) + r * dt
    c = -0.5 * (r * dt * veti + sigma**2 * dt * (veti**2))
    coeff = np.diag(a[2:M], -1) + np.diag(b[1:M]) + np.diag(c[1:M-1], 1)
    #lu, piv = lu_factor(coeff)
    
    if option_type == "Put":
        
        LU, piv = lu_factor(coeff)  # Equivalent to MATLAB's [L, U] = lu(coeff)

        # Solve the sequence of linear systems
        aux = np.zeros(M-1)

        for j in range(N-1, -1, -1):  # Reverse loop from N to 1
            aux[0] = -a[1] * matval[0, j]  # Adjust indexing for Python (0-based)
    
            # Solve L(Ux) = b using LU decomposition
            matval[1:M, j] = lu_solve((LU, piv), matval[1:M, j+1] + aux)
         
        price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
        price = price_interp(S0)
        
        return price, vetS, matval[:, 0]
    
    elif option_type == "Call":
        LU, piv = lu_factor(coeff)  # Equivalent to MATLAB's [L, U] = lu(coeff)

        # Solve the sequence of linear systems
        aux = np.zeros(M-1)

        for j in range(N-1, -1, -1):  # Reverse loop from N to 1
            aux[M-2] = -c[M-1] * matval[M, j]  # Adjust indexing for Python (0-based)
    
            # Solve L(Ux) = b using LU decomposition
            matval[1:M, j] = lu_solve((LU, piv), matval[1:M, j+1] + aux)
            
        price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
        price = price_interp(S0)

        
        return price, vetS, matval[:, 0]
    
def crank_nicolson(S0, K, r, T, sigma, dS, dt, option_type):
    Smax = 2*max(S0,K)*np.exp(r*T)
    M = round(Smax / dS)
    dS = Smax / M
    N = round(T / dt)
    dt = T / N
    matval = np.zeros((M+1, N+1))
    vetS = np.linspace(0, Smax, M+1)
    veti = np.arange(0, M+1)
    vetj = np.arange(0, N+1)

    if option_type == "Call":
        matval[:, -1] = np.maximum(vetS - K, 0)
        matval[0, :] = 0
        #matval[-1, :] = S_max - K * np.exp(-r * (T - np.linspace(0, T, N + 1)))
        matval[-1, :] = Smax - K * np.exp(-r * dt * (N - vetj))
    elif option_type == "Put":
        matval[:, -1] = np.maximum(K - vetS, 0)
        matval[0, :] = K * np.exp(-r * (T - np.linspace(0, T, N + 1)))
        matval[-1, :] = 0

    # Set up the coefficients matrix
    alpha = 0.25 * dt * (sigma**2 * (veti**2) - r * veti)
    beta = -0.5 * dt * (sigma**2 * (veti**2) + r)
    gamma = 0.25 * dt * (sigma**2 * (veti**2) + r * veti)

    # Construct tridiagonal matrices
    M1 = -np.diag(alpha[2:M], -1) + np.diag(1 - beta[1:M]) - np.diag(gamma[1:M-1], 1)
    M2 = np.diag(alpha[2:M], -1) + np.diag(1 + beta[1:M]) + np.diag(gamma[1:M-1], 1)

    # LU decomposition for efficient solving
    LU, piv = lu_factor(M1)

    # Solve the sequence of linear systems
    lostval = np.zeros(M2.shape[1])

    for j in range(N-1, -1, -1):
        if len(lostval) > 1:
            lostval[0] = alpha[1] * (matval[0, j] + matval[0, j+1])
            lostval[-1] = gamma[-1] * (matval[-1, j] + matval[-1, j+1])
        else:
            lostval = lostval[0] + lostval[-1]

        rhs = M2 @ matval[1:M, j+1] + lostval
        matval[1:M, j] = lu_solve((LU, piv), rhs) 
    
    price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
    price = price_interp(S0)
                
    return price, vetS, matval[:, 0]



# Streamlit interface
st.title("Comparison of different numerical schemes and the analytical solution")

 #S_max = st.sidebar.number_input("Maximum Stock Price (S_max)", value=200.0, step=1.0)
S0 = st.sidebar.number_input("Spot Price (S0)", value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0, step=0.1)
r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05, step=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01)
dS = st.sidebar.number_input("Stock Price Step (dS)", value=10.0, step=0.1)
dt = st.sidebar.number_input("Time Step (dt)", value=0.001, step=0.001)
option_type = st.sidebar.selectbox("Option Type", ("Call", "Put"))
numerical_method = st.sidebar.selectbox("Numerical method", ("Forward Euler", "Backward Euler", "Crank-Nicolson"))
     

if numerical_method == "Forward Euler":
    # Compute Forward Euler results
    price, S_grid, forward_euler_prices = forward_euler(S0, K, T, r, sigma, dS, dt, option_type)

    # Compute analytical Black-Scholes prices
    analytical_prices = black_scholes(S_grid, K, T, r, sigma, option_type)
    analytical_price = black_scholes(S0, K, T, r, sigma, option_type)

    # Find the index closest to S0
    #index_S0 = (np.abs(S_grid - S0)).argmin()

    # Create a DataFrame for comparison at S0
    df = pd.DataFrame({
        "Forward Euler Price": [np.round(price, 4)],
        "Analytical Price": [analytical_price],
        "Absolute Error": [np.abs(price - analytical_price)],
    })

    # Display the table for the spot price
    st.subheader("Option Price Comparison at Spot Price (S0)")
    st.table(df)

    # Plot the results
    st.subheader("Comparison of Prices Across All Stock Prices")
    fig = go.Figure()

    # Scatter plot for Forward Euler
    fig.add_trace(go.Scatter(
        x=S_grid, 
        y=forward_euler_prices, 
        mode="markers", 
        name="Forward Euler Prices",
        marker=dict(color="red", size=6)
    ))

    # Line plot for Analytical Black-Scholes
    fig.add_trace(go.Scatter(
        x=S_grid, 
        y=analytical_prices, 
        mode="lines", 
        name="Analytical Black-Scholes Prices",
        line=dict(color="blue", width=2)
    ))

    fig.update_layout(
        title="Option Prices: Forward Euler vs Analytical",
        xaxis_title="Stock Price (S)",
        yaxis_title="Option Price (V)",
        legend_title="Method",
        width=800,
        height=500
    )

    st.plotly_chart(fig)
    
elif numerical_method == "Backward Euler":
    # Compute Forward Euler results
    price, S_grid, backward_euler_prices = backward_euler(S0, K, r, T, sigma, dS, dt, option_type)

    # Compute analytical Black-Scholes prices
    analytical_prices = black_scholes(S_grid, K, T, r, sigma, option_type)
    analytical_price = black_scholes(S0, K, T, r, sigma, option_type)

    # Find the index closest to S0
    index_S0 = (np.abs(S_grid - S0)).argmin()

    # Create a DataFrame for comparison at S0
    df = pd.DataFrame({
        "Backward Euler Price": [np.round(price, 4)],
        "Analytical Price": [analytical_price],
        "Absolute Error": [np.abs(price - analytical_price)],
    })

    # Display the table for the spot price
    st.subheader("Option Price Comparison at Spot Price (S0)")
    st.table(df)

    # Plot the results
    st.subheader("Comparison of Prices Across All Stock Prices")
    fig = go.Figure()

    # Scatter plot for Forward Euler
    fig.add_trace(go.Scatter(
        x=S_grid, 
        y=backward_euler_prices, 
        mode="markers", 
        name="Backward Euler Prices",
        marker=dict(color="red", size=6)
    ))

    # Line plot for Analytical Black-Scholes
    fig.add_trace(go.Scatter(
        x=S_grid, 
        y=analytical_prices, 
        mode="lines", 
        name="Analytical Black-Scholes Prices",
        line=dict(color="blue", width=2)
    ))

    fig.update_layout(
        title="Option Prices: Backward Euler vs Analytical",
        xaxis_title="Stock Price (S)",
        yaxis_title="Option Price (V)",
        legend_title="Method",
        width=800,
        height=500
    )
    st.plotly_chart(fig)

elif numerical_method == "Crank-Nicolson":
    # Compute Forward Euler results
    price, S_grid, crank_nicolson_prices= crank_nicolson(S0, K, r, T, sigma, dS, dt, option_type)

    # Compute analytical Black-Scholes prices
    analytical_price = black_scholes(S0, K, T, r, sigma, option_type)
    analytical_prices = black_scholes(S_grid, K, T, r, sigma, option_type)
    # Find the index closest to S0
    index_S0 = (np.abs(S_grid - S0)).argmin()


    # Create a DataFrame for comparison at S0
    df = pd.DataFrame({
        "Crank Nicolson Price": [np.round(price, 4)],
        "Analytical Price": [analytical_price],
        "Absolute Error": [np.abs(price - analytical_price)],
    })

    # Display the table for the spot price
    st.subheader("Option Price Comparison at Spot Price (S0)")
    st.table(df)

    # Plot the results
    st.subheader("Comparison of Prices Across All Stock Prices")
    fig = go.Figure()

    # Scatter plot for Forward Euler
    fig.add_trace(go.Scatter(
        x=S_grid, 
        y=crank_nicolson_prices, 
        mode="markers", 
        name="Crank Nicolson Prices",
        marker=dict(color="red", size=6)
    ))

    #Line plot for Analytical Black-Scholes
    fig.add_trace(go.Scatter(
        x=S_grid, 
        y=analytical_prices, 
        mode="lines", 
        name="Analytical Black-Scholes Prices",
        line=dict(color="blue", width=2)
    ))

    fig.update_layout(
        title="Option Prices: Crank Nicolson vs Analytical",
        xaxis_title="Stock Price (S)",
        yaxis_title="Option Price (V)",
        legend_title="Method",
        width=800,
        height=500
    )
    st.plotly_chart(fig)



import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import interp1d
from scipy.linalg import lu
import time

# -----------------------------------------------------------
#   REPLACE THESE with your own definitions or imports:
# -----------------------------------------------------------
def black_scholes(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) + 1e-12)
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "Call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "Put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def forward_euler(S0, K, T, r, sigma, dS, dt, option_type):
        S_max = 2*max(S0,K)*np.exp(r*T)
        M = int(S_max / dS)
        N = int(T / dt)
        dS = S_max / M
        dt = T / N
        veti = np.arange(0, M + 1)
        vetj = np.arange(0, N + 1)

        matval = np.zeros((M + 1, N + 1))
        vetS = np.linspace(0, S_max, M + 1)

        # Boundary conditions
        if option_type == "Call":
            matval[:, -1] = np.maximum(vetS - K, 0)
            matval[0, :] = 0
            matval[-1, :] = S_max - K * np.exp(-r * (T - np.linspace(0, T, N + 1)))
            #matval[-1, :] = S_max - K * np.exp(-r * dt * (N - vetj))
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
        #matval[-1, :] = Smax - K * np.exp(-r * (N - vetj))
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
        #matval[-1, :] = Smax - K * np.exp(-r * (N - vetj))
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


# -----------------------------------------------------------
#   PAGE LAYOUT
# -----------------------------------------------------------
st.set_page_config(page_title="Numerical Scheme comparisons", layout="wide")
st.title("Comparison of Forward/Backward/Crank–Nicolson Methods")

# Sidebar for user inputs
st.sidebar.header("Option & FD Parameters")
option_type = st.sidebar.selectbox("Option Type", ["Call", "Put"])
K          = st.sidebar.number_input("Strike (K)", value=100.0, step=1.0)
T          = st.sidebar.number_input("Maturity (T, in years)", value=1.0, step=0.1)
r          = st.sidebar.number_input("Risk-free rate (r)", value=0.05, step=0.01)
sigma      = st.sidebar.number_input("Volatility (sigma)", value=0.2, step=0.01)

st.sidebar.header("Range of Spot Prices")
S_min      = st.sidebar.number_input("Minimum Spot (S)", value=80.0, step=1.0)
S_max      = st.sidebar.number_input("Maximum Spot (S)", value=120.0, step=1.0)
S_step     = st.sidebar.number_input("Spot increment", value=5.0, step=1.0)

st.sidebar.header("FD Mesh Choices")
# Possibly separate dt/dS for each scheme if you wish
dt_explicit = st.sidebar.number_input("dt (Explicit)", value=0.0001, step=0.0001, format="%.6f")
dS_explicit = st.sidebar.number_input("dS (Explicit)", value=1.0, step=0.1)

dt_implicit = st.sidebar.number_input("dt (Implicit)", value=0.001, step=0.0001, format="%.6f")
dS_implicit = st.sidebar.number_input("dS (Implicit)", value=0.5, step=0.1)

dt_CN       = st.sidebar.number_input("dt (Crank–Nicolson)", value=0.01, step=0.001, format="%.3f")
dS_CN       = st.sidebar.number_input("dS (Crank–Nicolson)", value=0.5, step=0.1)

# Make a list to store table rows
rows = []

# Iterate over the requested spot prices
spots = np.arange(S_min, S_max + 0.1, S_step)
for S0 in spots:
    # -----------------------------------------------------
    #   1) True / Analytical Price
    # -----------------------------------------------------
    true_price = black_scholes(S0, K, T, r, sigma, option_type)

    # -----------------------------------------------------
    #   2) Forward Euler (Explicit)
    # -----------------------------------------------------
    t0 = time.perf_counter()
    FE_value, S_grid_FE, FE_prices = forward_euler(S0, K, T, r, sigma, dS_explicit, dt_explicit, option_type)
    time_FE  = time.perf_counter() - t0

    err_FE = abs(FE_value - true_price)
    accuracy_FE = 0.0
    if true_price != 0:
        accuracy_FE = 100 * (1 - err_FE / true_price)

    # -----------------------------------------------------
    #   3) Backward Euler (Implicit)
    # -----------------------------------------------------
    t0 = time.perf_counter()
    BE_value, S_grid_BE, BE_prices = backward_euler(S0, K, r, T, sigma, dS_implicit, dt_implicit, option_type)
    time_BE  = time.perf_counter() - t0

    err_BE = abs(BE_value - true_price)
    accuracy_BE = 0.0
    if true_price != 0:
        accuracy_BE = 100 * (1 - err_BE / true_price)

    # -----------------------------------------------------
    #   4) Crank–Nicolson
    # -----------------------------------------------------
    t0 = time.perf_counter()
    CN_value, S_grid_CN, CN_prices = crank_nicolson(S0, K, r, T, sigma, dS_CN, dt_CN, option_type)
    time_CN = time.perf_counter() - t0

    err_CN = abs(CN_value - true_price)
    accuracy_CN = 0.0
    if true_price != 0:
        accuracy_CN = 100 * (1 - err_CN / true_price)

    # -----------------------------------------------------
    #   5) Prepare row
    # -----------------------------------------------------
    row = {
        "Spot": f"{S0:.2f}",
        "True Value": f"{true_price:.4f}",
        
        "Exp Value": f"{FE_value:.4f}",
        "Exp Accuracy": f"{accuracy_FE:.2f}%",
        #"Exp Time (s)": f"{time_FE:.4f}",
        
        "Imp Value": f"{BE_value:.4f}",
        "Imp Accuracy": f"{accuracy_BE:.2f}%",
        #"Imp Time (s)": f"{time_BE:.4f}",
        
        "CN Value": f"{CN_value:.4f}",
        "CN Accuracy": f"{accuracy_CN:.2f}%",
        #"CN Time (s)": f"{time_CN:.4f}",
    }
    rows.append(row)

# Once done, build a final DataFrame
df = pd.DataFrame(rows)

st.subheader("Comparison of Three Finite‐Difference Methods vs. Black–Scholes")
st.table(df)

df_styled = (
    df.style
      .set_properties(**{"background-color": "lightblue"}, subset=["Exp Value", "Exp Accuracy"])
      .set_properties(**{"background-color": "lightgreen"}, subset=["Imp Value", "Imp Accuracy"])
      .set_properties(**{"background-color": "lightyellow"}, subset=["CN Value", "CN Accuracy"])
      #.format("{:.4f}")  # Example format for numeric columns
)

# Then display it with st.dataframe:
st.dataframe(df_styled)

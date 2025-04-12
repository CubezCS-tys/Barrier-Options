# #MAIN
# import streamlit as st
# import numpy as np
# import pandas as pd
# from scipy.stats import norm
# import plotly.graph_objects as go
# from scipy.linalg import lu_factor, lu_solve
# from scipy.interpolate import interp1d
# from scipy.linalg import lu
# import time

# # Black-Scholes formula for analytical solution
# def black_scholes(S, K, T, r, sigma, option_type):
#     d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) + 1e-12)
#     d2 = d1 - sigma * np.sqrt(T)
#     if option_type == "Call":
#         price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
#     elif option_type == "Put":
#         price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
#     return price


# def forward_euler(S0, K, T, r, sigma, dS, dt, option_type):
#     """
#     Forward Euler PDE for a vanilla European call on [0, S_max].
#     Returns: (priceVan, S_grid, V0).
#     """
#     S_max = 2 * max(S0, K) * np.exp(r * T)
#     M = int(S_max / dS)
#     N = int(T / dt)
#     dS = S_max / M  # adjust
#     dt = T / N      # adjust

#     S_grid = np.linspace(0, S_max, M + 1)
#     V = np.zeros((N + 1, M + 1))

#     if option_type == "Call":
#         # Terminal payoff
#         V[-1, :] = np.maximum(S_grid - K, 0.0)

#         # Time array
#         t_arr = np.linspace(0, T, N + 1)

#         # Boundary conditions:
#         #   - at S=0: call is 0
#         #   - at S=S_max: call ~ S_max - K e^{-r tau}
#         for i in range(N + 1):
#             tau = T - t_arr[i]
#             V[i, 0]   = 0.0
#             V[i, -1]  = S_max - K * np.exp(-r * tau)
    
#     else: 
#             # Terminal payoff
#         V[-1, :] = np.maximum(K - S_grid, 0.0)

#         # Time array
#         t_arr = np.linspace(0, T, N + 1)

#         # Boundary conditions for a put:
#         #   - at S=0:  put is ~ K e^{-r tau}
#         #   - at S=S_max: put is ~ 0
#         for i in range(N + 1):
#             tau = T - t_arr[i]
#             V[i, 0]   = K * np.exp(-r * tau)  # deep in-the-money for a put
#             V[i, -1]  = 0.0

#     # PDE coefficients
#     j_arr = np.arange(M + 1)
#     a = 0.5 * dt * (sigma**2 * j_arr**2 - r * j_arr)
#     b = 1.0 - dt * (sigma**2 * j_arr**2 + r)
#     c = 0.5 * dt * (sigma**2 * j_arr**2 + r * j_arr)

#     # Forward Euler stepping: from n=N down to n=1
#     for n in range(N, 0, -1):
#         for j in range(1, M):
#             V[n - 1, j] = a[j] * V[n, j - 1] + b[j] * V[n, j] + c[j] * V[n, j + 1]

#     # Interpolate to get the price at S0
#     interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
#     priceVan = float(interp_fn(S0))
#     return priceVan, S_grid, V[0, :]



# def backward_euler(S0, K, r, T, sigma, dS, dt, option_type):
#     """
#     Backward Euler PDE for a vanilla European option on [0, S_max].
#     Returns: (priceVan, S_grid, V_at_t0).
#     Handles both Call and Put options.
#     """
#     # 1) Grid setup
#     S_max = 2 * max(S0, K) * np.exp(r * T)
#     M = int(S_max / dS)
#     N = int(T / dt)
#     dS = S_max / M
#     dt = T / N

#     S_grid = np.linspace(0, S_max, M + 1)
#     V = np.zeros((N + 1, M + 1))
    
#     # 2) Terminal payoff
#     if option_type == "Call":
#         V[-1, :] = np.maximum(S_grid - K, 0.0)
#     else:  # Put option
#         V[-1, :] = np.maximum(K - S_grid, 0.0)

#     # 3) PDE coefficients for the implicit scheme (same for Call and Put)
#     j_arr = np.arange(M + 1)
#     A_ = -0.5 * dt * (sigma**2 * j_arr**2 - r * j_arr)
#     B_ =  1.0 + dt * (sigma**2 * j_arr**2 + r)
#     C_ = -0.5 * dt * (sigma**2 * j_arr**2 + r * j_arr)

#     # Tridiagonal matrix for interior nodes j = 1,...,M-1
#     main_diag = B_[1:M]
#     lower_diag = A_[2:M]     # subdiagonal
#     upper_diag = C_[1:M-1]   # superdiagonal
#     T_mat = np.diag(main_diag)
#     if M - 2 > 0:
#         T_mat += np.diag(lower_diag, k=-1) + np.diag(upper_diag, k=1)
#     else:
#         T_mat = T_mat.reshape((1, 1))

#     # 4) Time-stepping from n = N down to 1
#     t_arr = np.linspace(0, T, N + 1)
#     for n in range(N, 0, -1):
#         tau = T - t_arr[n - 1]
#         # Apply boundary conditions according to option type:
#         if option_type == "Call":
#             V[n - 1, 0]   = 0.0
#             V[n - 1, -1]  = S_max - K * np.exp(-r * tau)
#         else:  # Put option
#             V[n - 1, 0]   = K * np.exp(-r * tau)
#             V[n - 1, -1]  = 0.0

#         # Right-hand side from V^n; adjust for boundaries:
#         rhs = V[n, 1:M].copy()
#         rhs[0]   -= A_[1] * V[n - 1, 0]
#         rhs[-1]  -= C_[M - 1] * V[n - 1, -1]

#         # Solve the linear system:
#         V[n - 1, 1:M] = np.linalg.solve(T_mat, rhs)

#     # 5) Interpolate to get price at S0
#     interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
#     price = float(interp_fn(S0))
#     return price, S_grid, V[0, :]


# def crank_nicolson(S0, K, r, T, sigma, dS, dt, option_type):
#     """
#     Crank–Nicolson PDE for a vanilla European option on [0, S_max].
#     Returns: (price, S_grid, V_at_t0).
#     Handles both Call and Put options.
#     """
#     # 1) Grid setup
#     S_max = 2 * max(S0, K) * np.exp(r * T)
#     M = int(S_max / dS)
#     N = int(T / dt)
#     dS = S_max / M  # adjust
#     dt = T / N      # adjust

#     S_grid = np.linspace(0, S_max, M + 1)
#     V = np.zeros((N + 1, M + 1))
    
#     # 2) Terminal payoff
#     if option_type == "Call":
#         V[-1, :] = np.maximum(S_grid - K, 0.0)
#     else:  # Put option
#         V[-1, :] = np.maximum(K - S_grid, 0.0)
    
#     # 3) Precompute Crank–Nicolson coefficients for j = 0,...,M.
#     # Here we define:
#     #   a[j] = 0.25 * dt * (sigma**2 * j**2 - r * j)
#     #   b[j] = 0.5  * dt * (sigma**2 * j**2 + r)
#     #   c[j] = 0.25 * dt * (sigma**2 * j**2 + r * j)
#     j_arr = np.arange(M + 1)
#     a = 0.25 * dt * (sigma**2 * j_arr**2 - r * j_arr)
#     b = 0.5  * dt * (sigma**2 * j_arr**2 + r)
#     c = 0.25 * dt * (sigma**2 * j_arr**2 + r * j_arr)
    
#     # Build the tridiagonal matrix for interior nodes j = 1,...,M-1.
#     main_diag = 1 + b[1:M]
#     lower_diag = -a[2:M]      # corresponds to V_{j-1}^{n-1} for j = 2,...,M-1
#     upper_diag = -c[1:M-1]    # corresponds to V_{j+1}^{n-1} for j = 1,...,M-2
#     LHS = np.diag(main_diag)
#     if M - 2 > 0:
#         LHS += np.diag(lower_diag, k=-1) + np.diag(upper_diag, k=1)
    
#     # 4) Time-stepping (backward in time)
#     t_arr = np.linspace(0, T, N + 1)
#     for n in range(N, 0, -1):
#         tau = T - t_arr[n - 1]
#         # Apply boundary conditions according to option type:
#         if option_type == "Call":
#             V[n - 1, 0]   = 0.0
#             V[n - 1, -1]  = S_max - K * np.exp(-r * tau)
#         else:  # Put option
#             V[n - 1, 0]   = K * np.exp(-r * tau)
#             V[n - 1, -1]  = 0.0
        
#         # Build right-hand side for interior nodes j = 1,...,M-1.
#         rhs = (a[1:M] * V[n, 0:M-1] +
#                (1 - b[1:M]) * V[n, 1:M] +
#                c[1:M] * V[n, 2:M+1])
#         # Adjust for known boundary values:
#         rhs[0]   += a[1] * V[n - 1, 0]
#         rhs[-1]  += c[M - 1] * V[n - 1, -1]
        
#         # Solve for interior nodes:
#         V[n - 1, 1:M] = np.linalg.solve(LHS, rhs)
    
#     # 5) Interpolate to get the price at S0:
#     interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
#     price = float(interp_fn(S0))
#     return price, S_grid, V[0, :]




# # Streamlit interface
# st.title("Comparison of different numerical schemes and the analytical solution")

#  #S_max = st.sidebar.number_input("Maximum Stock Price (S_max)", value=200.0, step=1.0)
# S0 = st.sidebar.number_input("Spot Price (S0)", value=100.0, step=1.0)
# K = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0)
# T = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0, step=0.1)
# r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05, step=0.01)
# sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01)
# dS = st.sidebar.number_input("Stock Price Step (dS)", value=10.0, step=0.1)
# dt = st.sidebar.number_input("Time Step (dt)", value=0.001, step=0.001)
# option_type = st.sidebar.selectbox("Option Type", ("Call", "Put"))
# numerical_method = st.sidebar.selectbox("Numerical method", ("Forward Euler", "Backward Euler", "Crank-Nicolson"))
     

# if numerical_method == "Forward Euler":
#     # Compute Forward Euler results
#     price, S_grid, forward_euler_prices = forward_euler(S0, K, T, r, sigma, dS, dt, option_type)

#     # Compute analytical Black-Scholes prices
#     analytical_prices = black_scholes(S_grid, K, T, r, sigma, option_type)
#     analytical_price = black_scholes(S0, K, T, r, sigma, option_type)

#     # Find the index closest to S0
#     #index_S0 = (np.abs(S_grid - S0)).argmin()

#     # Create a DataFrame for comparison at S0
#     df = pd.DataFrame({
#         "Forward Euler Price": [np.round(price, 4)],
#         "Analytical Price": [analytical_price],
#         "Absolute Error": [np.abs(price - analytical_price)],
#     })

#     # Display the table for the spot price
#     st.subheader("Option Price Comparison at Spot Price (S0)")
#     st.table(df)

#     # Plot the results
#     st.subheader("Comparison of Prices Across All Stock Prices")
#     fig = go.Figure()

#     # Scatter plot for Forward Euler
#     fig.add_trace(go.Scatter(
#         x=S_grid, 
#         y=forward_euler_prices, 
#         mode="markers", 
#         name="Forward Euler Prices",
#         marker=dict(color="red", size=6)
#     ))

#     # Line plot for Analytical Black-Scholes
#     fig.add_trace(go.Scatter(
#         x=S_grid, 
#         y=analytical_prices, 
#         mode="lines", 
#         name="Analytical Black-Scholes Prices",
#         line=dict(color="blue", width=2)
#     ))

#     fig.update_layout(
#         title="Option Prices: Forward Euler vs Analytical",
#         xaxis_title="Stock Price (S)",
#         yaxis_title="Option Price (V)",
#         legend_title="Method",
#         width=800,
#         height=500
#     )

#     st.plotly_chart(fig)
    
# elif numerical_method == "Backward Euler":
#     # Compute Forward Euler results
#     price, S_grid, backward_euler_prices = backward_euler(S0, K, r, T, sigma, dS, dt, option_type)

#     # Compute analytical Black-Scholes prices
#     analytical_prices = black_scholes(S_grid, K, T, r, sigma, option_type)
#     analytical_price = black_scholes(S0, K, T, r, sigma, option_type)

#     # Find the index closest to S0
#     index_S0 = (np.abs(S_grid - S0)).argmin()

#     # Create a DataFrame for comparison at S0
#     df = pd.DataFrame({
#         "Backward Euler Price": [np.round(price, 4)],
#         "Analytical Price": [analytical_price],
#         "Absolute Error": [np.abs(price - analytical_price)],
#     })

#     # Display the table for the spot price
#     st.subheader("Option Price Comparison at Spot Price (S0)")
#     st.table(df)

#     # Plot the results
#     st.subheader("Comparison of Prices Across All Stock Prices")
#     fig = go.Figure()

#     # Scatter plot for Forward Euler
#     fig.add_trace(go.Scatter(
#         x=S_grid, 
#         y=backward_euler_prices, 
#         mode="markers", 
#         name="Backward Euler Prices",
#         marker=dict(color="red", size=6)
#     ))

#     # Line plot for Analytical Black-Scholes
#     fig.add_trace(go.Scatter(
#         x=S_grid, 
#         y=analytical_prices, 
#         mode="lines", 
#         name="Analytical Black-Scholes Prices",
#         line=dict(color="blue", width=2)
#     ))

#     fig.update_layout(
#         title="Option Prices: Backward Euler vs Analytical",
#         xaxis_title="Stock Price (S)",
#         yaxis_title="Option Price (V)",
#         legend_title="Method",
#         width=800,
#         height=500
#     )
#     st.plotly_chart(fig)

# elif numerical_method == "Crank-Nicolson":
#     # Compute Forward Euler results
#     price, S_grid, crank_nicolson_prices= crank_nicolson(S0, K, r, T, sigma, dS, dt, option_type)

#     # Compute analytical Black-Scholes prices
#     analytical_price = black_scholes(S0, K, T, r, sigma, option_type)
#     analytical_prices = black_scholes(S_grid, K, T, r, sigma, option_type)
#     # Find the index closest to S0
#     index_S0 = (np.abs(S_grid - S0)).argmin()


#     # Create a DataFrame for comparison at S0
#     df = pd.DataFrame({
#         "Crank Nicolson Price": [np.round(price, 4)],
#         "Analytical Price": [analytical_price],
#         "Absolute Error": [np.abs(price - analytical_price)],
#     })

#     # Display the table for the spot price
#     st.subheader("Option Price Comparison at Spot Price (S0)")
#     st.table(df)

#     # Plot the results
#     st.subheader("Comparison of Prices Across All Stock Prices")
#     fig = go.Figure()

#     # Scatter plot for Forward Euler
#     fig.add_trace(go.Scatter(
#         x=S_grid, 
#         y=crank_nicolson_prices, 
#         mode="markers", 
#         name="Crank Nicolson Prices",
#         marker=dict(color="red", size=6)
#     ))

#     #Line plot for Analytical Black-Scholes
#     fig.add_trace(go.Scatter(
#         x=S_grid, 
#         y=analytical_prices, 
#         mode="lines", 
#         name="Analytical Black-Scholes Prices",
#         line=dict(color="blue", width=2)
#     ))

#     fig.update_layout(
#         title="Option Prices: Crank Nicolson vs Analytical",
#         xaxis_title="Stock Price (S)",
#         yaxis_title="Option Price (V)",
#         legend_title="Method",
#         width=800,
#         height=500
#     )
#     st.plotly_chart(fig)

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go
from scipy.interpolate import interp1d
st.set_page_config("Finite Pricing Vanilla Options", layout="wide")
# -----------------------------------------------------------------------------
# Custom CSS for enhanced styling including metric boxes
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* General background and font */
    body {
      background-color: #f8f9fa;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Main container styling */
    .reportview-container .main {
        background-color: #ffffff;
        padding: 1rem 2rem;
        border-radius: 10px;
    }
    /* Sidebar styling */
    .css-1d391kg {  
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    /* Style the input fields in Streamlit */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input {
        border-radius: 5px;
        border: 1px solid #aaa;
    }
    /* Metric cards container */
    .metric-container {
        display: flex;
        flex-direction: row;
        justify-content: space-evenly;
        margin-bottom: 1rem;
    }
    /* Individual metric card style */
    .metric-card {
        background-color: #fafafa;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        flex: 1;
        margin: 0 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 1rem;
        font-weight: 600;
        color: #555;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 0.2rem;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Option pricing functions (analytical & finite differences)
# -----------------------------------------------------------------------------
def black_scholes(S, K, T, r, sigma, option_type):
    """Black-Scholes formula for European Call/Put options."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T) + 1e-12)
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "Call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # Put option
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def forward_euler(S0, K, T, r, sigma, dS, dt, option_type):
    """
    Forward Euler scheme (explicit) using finite differences.
    Handles both Call and Put options.
    
    Includes a stability warning: dt should be less than dS²/(sigma² * S_max²).
    """
    S_max = 2 * max(S0, K) * np.exp(r * T)
    M = int(S_max / dS)
    N = int(T / dt)
    dS = S_max / M  # adjust grid step
    dt = T / N      # adjust time step

    # Check stability condition for the explicit scheme:
    stability_limit = dS ** 2 / (sigma ** 2 * S_max ** 2)
    if dt > stability_limit:
        st.sidebar.warning(
            f"Warning: dt = {dt:.2e} exceeds the stability limit {stability_limit:.2e}. Consider using a smaller dt."
        )

    S_grid = np.linspace(0, S_max, M + 1)
    V = np.zeros((N + 1, M + 1))

    # Terminal payoff
    if option_type == "Call":
        V[-1, :] = np.maximum(S_grid - K, 0.0)
    else:
        V[-1, :] = np.maximum(K - S_grid, 0.0)

    t_arr = np.linspace(0, T, N + 1)
    # Set boundary conditions for each time step:
    for i in range(N + 1):
        tau = T - t_arr[i]
        if option_type == "Call":
            V[i, 0] = 0.0
            V[i, -1] = S_max - K * np.exp(-r * tau)
        else:
            V[i, 0] = K * np.exp(-r * tau)
            V[i, -1] = 0.0

    # Finite difference coefficients (using half weight for the convection term)
    j_arr = np.arange(M + 1)
    a = dt * (0.5 * sigma ** 2 * j_arr ** 2 - 0.5 * r * j_arr)
    b = 1.0 - dt * (sigma ** 2 * j_arr ** 2 + r)
    c = dt * (0.5 * sigma ** 2 * j_arr ** 2 + 0.5 * r * j_arr)

    # Time-stepping backward in time
    for n in range(N, 0, -1):
        for j in range(1, M):
            V[n - 1, j] = a[j] * V[n, j - 1] + b[j] * V[n, j] + c[j] * V[n, j + 1]

    interp_fn = interp1d(S_grid, V[0, :], kind="linear", fill_value="extrapolate")
    price = float(interp_fn(S0))
    return price, S_grid, V[0, :]

def backward_euler(S0, K, r, T, sigma, dS, dt, option_type):
    """
    Backward Euler scheme (implicit) using finite differences.
    Handles both Call and Put options.
    """
    S_max = 2 * max(S0, K) * np.exp(r * T)
    M = int(S_max / dS)
    N = int(T / dt)
    dS = S_max / M
    dt = T / N

    S_grid = np.linspace(0, S_max, M + 1)
    V = np.zeros((N + 1, M + 1))
    
    if option_type == "Call":
        V[-1, :] = np.maximum(S_grid - K, 0.0)
    else:
        V[-1, :] = np.maximum(K - S_grid, 0.0)

    j_arr = np.arange(M + 1)
    A_ = -0.5 * dt * (sigma ** 2 * j_arr ** 2 - r * j_arr)
    B_ = 1.0 + dt * (sigma ** 2 * j_arr ** 2 + r)
    C_ = -0.5 * dt * (sigma ** 2 * j_arr ** 2 + r * j_arr)

    main_diag = B_[1:M]
    lower_diag = A_[2:M]
    upper_diag = C_[1:M - 1]
    T_mat = np.diag(main_diag)
    if M - 2 > 0:
        T_mat += np.diag(lower_diag, k=-1) + np.diag(upper_diag, k=1)
    else:
        T_mat = T_mat.reshape((1, 1))

    t_arr = np.linspace(0, T, N + 1)
    for n in range(N, 0, -1):
        tau = T - t_arr[n - 1]
        if option_type == "Call":
            V[n - 1, 0] = 0.0
            V[n - 1, -1] = S_max - K * np.exp(-r * tau)
        else:
            V[n - 1, 0] = K * np.exp(-r * tau)
            V[n - 1, -1] = 0.0

        rhs = V[n, 1:M].copy()
        rhs[0] -= A_[1] * V[n - 1, 0]
        rhs[-1] -= C_[M - 1] * V[n - 1, -1]
        V[n - 1, 1:M] = np.linalg.solve(T_mat, rhs)

    interp_fn = interp1d(S_grid, V[0, :], kind="linear", fill_value="extrapolate")
    price = float(interp_fn(S0))
    return price, S_grid, V[0, :]

def crank_nicolson(S0, K, r, T, sigma, dS, dt, option_type):
    """
    Crank–Nicolson scheme using finite differences.
    Handles both Call and Put options.
    """
    S_max = 2 * max(S0, K) * np.exp(r * T)
    M = int(S_max / dS)
    N = int(T / dt)
    dS = S_max / M  
    dt = T / N      

    S_grid = np.linspace(0, S_max, M + 1)
    V = np.zeros((N + 1, M + 1))
    
    if option_type == "Call":
        V[-1, :] = np.maximum(S_grid - K, 0.0)
    else:
        V[-1, :] = np.maximum(K - S_grid, 0.0)
    
    j_arr = np.arange(M + 1)
    a = 0.25 * dt * (sigma ** 2 * j_arr ** 2 - r * j_arr)
    b = 0.5 * dt * (sigma ** 2 * j_arr ** 2 + r)
    c = 0.25 * dt * (sigma ** 2 * j_arr ** 2 + r * j_arr)
    
    main_diag = 1 + b[1:M]
    lower_diag = -a[2:M]
    upper_diag = -c[1:M - 1]
    LHS = np.diag(main_diag)
    if M - 2 > 0:
        LHS += np.diag(lower_diag, k=-1) + np.diag(upper_diag, k=1)
    
    t_arr = np.linspace(0, T, N + 1)
    for n in range(N, 0, -1):
        tau = T - t_arr[n - 1]
        if option_type == "Call":
            V[n - 1, 0] = 0.0
            V[n - 1, -1] = S_max - K * np.exp(-r * tau)
        else:
            V[n - 1, 0] = K * np.exp(-r * tau)
            V[n - 1, -1] = 0.0
        
        rhs = (a[1:M] * V[n, 0:M - 1] +
               (1 - b[1:M]) * V[n, 1:M] +
               c[1:M] * V[n, 2:M + 1])
        rhs[0] += a[1] * V[n - 1, 0]
        rhs[-1] += c[M - 1] * V[n - 1, -1]
        V[n - 1, 1:M] = np.linalg.solve(LHS, rhs)
    
    interp_fn = interp1d(S_grid, V[0, :], kind="linear", fill_value="extrapolate")
    price = float(interp_fn(S0))
    return price, S_grid, V[0, :]

# -----------------------------------------------------------------------------
# Streamlit UI: Finite Difference Scheme Page with Side-by-Side Boxes & Graphs
# -----------------------------------------------------------------------------

st.title("Option Pricing: Finite Difference Schemes vs. Analytical Solution")
st.markdown(
    "Select the numerical scheme and input parameters in the sidebar. "
    "Below, the computed prices are shown in side-by-side boxes and two graphs are displayed side by side."
)

# Sidebar inputs
st.sidebar.header("Input Parameters")
S0 = st.sidebar.number_input("Spot Price (S0)", value=100.0, step=1.0, format="%.2f")
K = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0, format="%.2f")
T = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0, step=0.1, format="%.2f")
r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05, step=0.01, format="%.4f")
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01, format="%.4f")
dS = st.sidebar.number_input("Stock Price Step (ΔS)", value=10.0, step=0.1, format="%.2f")
dt = st.sidebar.number_input("Time Step (Δt)", value=0.001, step=0.001, format="%.4f")
option_type = st.sidebar.selectbox("Option Type", ("Call", "Put"))
numerical_method = st.sidebar.selectbox("Numerical Method", ("Forward Euler", "Backward Euler", "Crank-Nicolson"))

# Compute option price using the selected numerical method
if numerical_method == "Forward Euler":
    price, S_grid, fd_prices = forward_euler(S0, K, T, r, sigma, dS, dt, option_type)
elif numerical_method == "Backward Euler":
    price, S_grid, fd_prices = backward_euler(S0, K, r, T, sigma, dS, dt, option_type)
elif numerical_method == "Crank-Nicolson":
    price, S_grid, fd_prices = crank_nicolson(S0, K, r, T, sigma, dS, dt, option_type)
else:
    st.error("Select a valid numerical scheme")
    st.stop()

# Compute analytical price at S0 and across S_grid
analytical_price = black_scholes(S0, K, T, r, sigma, option_type)
analytical_prices = black_scholes(S_grid, K, T, r, sigma, option_type)
abs_error = np.abs(price - analytical_price)

# -----------------------------------------------------------------------------
# Display computed prices in custom metric boxes (side by side)
# -----------------------------------------------------------------------------
st.markdown('<div class="metric-container">', unsafe_allow_html=True)
colA, colB, colC = st.columns(3)
with colA:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">{numerical_method} Price</div>
        <div class="metric-value">${price:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with colB:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">Analytical Price</div>
        <div class="metric-value">${analytical_price:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with colC:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">Absolute Error</div>
        <div class="metric-value">${abs_error:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Display two plots side by side using st.columns
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)

# Plot 1: Numerical vs. Analytical Option Prices
with col1:
    st.subheader("Option Prices vs. Stock Price")
    fig_prices = go.Figure()
    fig_prices.add_trace(go.Scatter(
        x=S_grid,
        y=fd_prices,
        mode="markers",
        name=f"{numerical_method} Prices",
        marker=dict(color="red", size=6)
    ))
    fig_prices.add_trace(go.Scatter(
        x=S_grid,
        y=analytical_prices,
        mode="lines",
        name="Analytical Prices",
        line=dict(color="blue", width=2)
    ))
    fig_prices.update_layout(
        xaxis_title="Stock Price (S)",
        yaxis_title="Option Price (V)",
        legend_title="Method",
        width=400,
        height=400
    )
    st.plotly_chart(fig_prices, use_container_width=True)

# Plot 2: Absolute Error vs. Stock Price
with col2:
    st.subheader("Absolute Error vs. Stock Price")
    error_values = np.abs(fd_prices - analytical_prices)
    fig_error = go.Figure()
    fig_error.add_trace(go.Scatter(
        x=S_grid,
        y=error_values,
        mode="lines+markers",
        line=dict(color="green", width=2),
        marker=dict(size=6),
        name="Absolute Error"
    ))
    fig_error.update_layout(
        xaxis_title="Stock Price (S)",
        yaxis_title="Absolute Error",
        width=400,
        height=400
    )
    st.plotly_chart(fig_error, use_container_width=True)



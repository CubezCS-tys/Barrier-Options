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

def forward_euler(S0, K, T, r, sigma, dS, dt, option_type):
    """
    Forward Euler PDE for a vanilla European call on [0, S_max].
    Returns: (priceVan, S_grid, V0).
    """
    S_max = 2 * max(S0, K) * np.exp(r * T)
    M = int(S_max / dS)
    N = int(T / dt)
    dS = S_max / M  # adjust
    dt = T / N      # adjust

    S_grid = np.linspace(0, S_max, M + 1)
    V = np.zeros((N + 1, M + 1))

    if option_type == "Call":
        # Terminal payoff
        V[-1, :] = np.maximum(S_grid - K, 0.0)

        # Time array
        t_arr = np.linspace(0, T, N + 1)

        # Boundary conditions:
        #   - at S=0: call is 0
        #   - at S=S_max: call ~ S_max - K e^{-r tau}
        for i in range(N + 1):
            tau = T - t_arr[i]
            V[i, 0]   = 0.0
            V[i, -1]  = S_max - K * np.exp(-r * tau)
    
    else: 
            # Terminal payoff
        V[-1, :] = np.maximum(K - S_grid, 0.0)

        # Time array
        t_arr = np.linspace(0, T, N + 1)

        # Boundary conditions for a put:
        #   - at S=0:  put is ~ K e^{-r tau}
        #   - at S=S_max: put is ~ 0
        for i in range(N + 1):
            tau = T - t_arr[i]
            V[i, 0]   = K * np.exp(-r * tau)  # deep in-the-money for a put
            V[i, -1]  = 0.0

    # PDE coefficients
    j_arr = np.arange(M + 1)
    a = 0.5 * dt * (sigma**2 * j_arr**2 - r * j_arr)
    b = 1.0 - dt * (sigma**2 * j_arr**2 + r)
    c = 0.5 * dt * (sigma**2 * j_arr**2 + r * j_arr)

    # Forward Euler stepping: from n=N down to n=1
    for n in range(N, 0, -1):
        for j in range(1, M):
            V[n - 1, j] = a[j] * V[n, j - 1] + b[j] * V[n, j] + c[j] * V[n, j + 1]

    # Interpolate to get the price at S0
    interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
    priceVan = float(interp_fn(S0))
    return priceVan, S_grid, V[0, :]



def backward_euler(S0, K, r, T, sigma, dS, dt, option_type):
    """
    Backward Euler PDE for a vanilla European call on [0, S_max].
    Returns: (priceVan, S_grid, V_at_t0).
    """
    # 1) Grid setup
    S_max = 2 * max(S0, K) * np.exp(r * T)
    M = int(S_max / dS)
    N = int(T / dt)
    dS = S_max / M
    dt = T / N

    S_grid = np.linspace(0, S_max, M + 1)
    V = np.zeros((N + 1, M + 1))

    # 2) Terminal payoff
    V[-1, :] = np.maximum(S_grid - K, 0.0)

    # 3) PDE coefficients for the implicit scheme
    j_arr = np.arange(M + 1)
    A_ = -0.5 * dt * (sigma**2 * j_arr**2 - r * j_arr)
    B_ =  1.0 + dt * (sigma**2 * j_arr**2 + r)
    C_ = -0.5 * dt * (sigma**2 * j_arr**2 + r * j_arr)

    # Tridiagonal matrix for j=1,...,M-1
    main_diag = B_[1:M]
    lower_diag = A_[2:M]     # subdiagonal
    upper_diag = C_[1:M-1]   # superdiagonal
    T_mat = np.diag(main_diag)
    if M - 2 > 0:
        T_mat += np.diag(lower_diag, k=-1) + np.diag(upper_diag, k=1)
    else:
        T_mat = T_mat.reshape((1, 1))

    # 4) Time-stepping from n=N down to 1
    t_arr = np.linspace(0, T, N + 1)
    for n in range(N, 0, -1):
        tau = T - t_arr[n - 1]
        # Boundary conditions at time level (n-1)
        V[n - 1, 0]   = 0.0
        V[n - 1, -1]  = S_max - K * np.exp(-r * tau)

        # Right-hand side from V^n
        rhs = V[n, 1:M].copy()
        # Adjust for known boundaries
        rhs[0]   -= A_[1]     * V[n - 1, 0]
        rhs[-1]  -= C_[M - 1] * V[n - 1, -1]

        # Solve the linear system
        V[n - 1, 1:M] = np.linalg.solve(T_mat, rhs)

    # 5) Interpolate to get price at S0
    interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
    price = float(interp_fn(S0))
    return price, S_grid, V[0, :]

    
def crank_nicolson(S0, K, r, T, sigma, dS, dt, option_type):
    """
    Crank–Nicolson PDE for a vanilla European call on [0, S_max].
    Returns: (price, S_grid, V_at_t0).
    """
    S_max = 2 * max(S0, K) * np.exp(r * T)
    M = int(S_max / dS)
    N = int(T / dt)
    dS = S_max / M  # adjust
    dt = T / N      # adjust

    S_grid = np.linspace(0, S_max, M + 1)
    V = np.zeros((N + 1, M + 1))
    # Terminal payoff:
    V[-1, :] = np.maximum(S_grid - K, 0.0)
    
    # Precompute Crank–Nicolson coefficients for j = 0,...,M.
    # Here we define:
    #   a[j] = 0.25 * dt * (sigma**2 * j**2 - r * j)
    #   b[j] = 0.5  * dt * (sigma**2 * j**2 + r)
    #   c[j] = 0.25 * dt * (sigma**2 * j**2 + r * j)
    j_arr = np.arange(M + 1)
    a = 0.25 * dt * (sigma**2 * j_arr**2 - r * j_arr)
    b = 0.5  * dt * (sigma**2 * j_arr**2 + r)
    c = 0.25 * dt * (sigma**2 * j_arr**2 + r * j_arr)
    
    # Build the tridiagonal matrix for interior nodes j = 1,...,M-1.
    main_diag = 1 + b[1:M]
    lower_diag = -a[2:M]      # corresponds to V_{j-1}^{n-1} for j = 2,...,M-1
    upper_diag = -c[1:M-1]    # corresponds to V_{j+1}^{n-1} for j = 1,...,M-2
    LHS = np.diag(main_diag)
    if M - 2 > 0:
        LHS += np.diag(lower_diag, k=-1) + np.diag(upper_diag, k=1)
    
    # Time-stepping (backward in time)
    t_arr = np.linspace(0, T, N + 1)
    for n in range(N, 0, -1):
        tau = T - t_arr[n - 1]
        # Boundary conditions for a call:
        V[n - 1, 0]   = 0.0
        V[n - 1, -1]  = S_max - K * np.exp(-r * tau)
        
        # Build right-hand side for interior nodes j = 1,...,M-1.
        # Using the explicit part:
        #   rhs[j-1] = a[j]*V[n, j-1] + (1 - b[j])*V[n, j] + c[j]*V[n, j+1]
        rhs = a[1:M] * V[n, 0:M-1] + (1 - b[1:M]) * V[n, 1:M] + c[1:M] * V[n, 2:M+1]
        # Adjust for known boundary values:
        # For j = 1 (leftmost interior): add a[1]*V[n-1,0] (V[n-1,0] is already set)
        rhs[0]   += a[1] * V[n - 1, 0]
        # For j = M-1: add c[M-1]*V[n-1,-1]
        rhs[-1]  += c[M - 1] * V[n - 1, -1]
        
        # Solve for interior nodes:
        V[n - 1, 1:M] = np.linalg.solve(LHS, rhs)
    
    # Interpolate to get the price at S0:
    interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
    price = float(interp_fn(S0))
    return price, S_grid, V[0, :]



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


# st.set_page_config(page_title="FDM vs Analytical", layout="wide")
# st.title("📊 Comparison of Numerical Schemes vs Analytical Black-Scholes")

# # Sidebar Inputs
# st.sidebar.header("🛠️ Input Parameters")
# S0 = st.sidebar.number_input("Spot Price (S0)", value=100.0, step=1.0)
# K = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0)
# T = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0, step=0.1)
# r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05, step=0.01)
# sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01)
# dS = st.sidebar.number_input("Stock Price Step (dS)", value=10.0, step=0.1)
# dt = st.sidebar.number_input("Time Step (dt)", value=0.001, step=0.001)
# option_type = st.sidebar.selectbox("Option Type", ("Call", "Put"))
# numerical_method = st.sidebar.selectbox("Numerical method", ("Forward Euler", "Backward Euler", "Crank-Nicolson"))

# # Placeholder to call the correct method dynamically
# scheme_func = {
#     "Forward Euler": forward_euler,
#     "Backward Euler": backward_euler,
#     "Crank-Nicolson": crank_nicolson
# }

# # Run selected scheme
# price, S_grid, scheme_prices = scheme_func[numerical_method](S0, K, r, T, sigma, dS, dt, option_type)
# analytical_prices = black_scholes(S_grid, K, T, r, sigma, option_type)
# analytical_price = black_scholes(S0, K, T, r, sigma, option_type)

# # Two column layout
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader(f"📍 {numerical_method} vs Analytical at S₀ = {S0}")
#     df = pd.DataFrame({
#         f"{numerical_method} Price": [np.round(price, 4)],
#         "Analytical Price": [np.round(analytical_price, 4)],
#         "Absolute Error": [np.round(np.abs(price - analytical_price), 4)]
#     })
#     st.table(df)

# with col2:
#     st.subheader("📈 Comparison of Prices Across Grid")
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=S_grid, y=scheme_prices, mode='markers', name=f'{numerical_method} Prices', marker=dict(color='crimson')))
#     fig.add_trace(go.Scatter(x=S_grid, y=analytical_prices, mode='lines', name='Analytical Black-Scholes', line=dict(color='royalblue')))
#     fig.update_layout(
#         xaxis_title="Stock Price (S)",
#         yaxis_title="Option Price (V)",
#         legend_title="Method",
#         height=500
#     )
#     st.plotly_chart(fig, use_container_width=True)

# # Expandable error plot
# with st.expander("🔍 View Absolute Error across Stock Grid"):
#     fig_err = go.Figure()
#     fig_err.add_trace(go.Scatter(x=S_grid, y=np.abs(scheme_prices - analytical_prices), mode='lines+markers', name="|Error|", line=dict(color='orange')))
#     fig_err.update_layout(
#         xaxis_title="Stock Price (S)",
#         yaxis_title="Absolute Error",
#         height=400
#     )
#     st.plotly_chart(fig_err, use_container_width=True)

# st.set_page_config(page_title="Option Pricing Comparison", layout="wide")
# st.title("Comparison of Numerical Schemes for Option Pricing")

# # ------------------------------
# # Custom CSS for Info Boxes
# # ------------------------------
# st.markdown(
#     """
#     <style>
#     .info-box {
#         background-color: #f9f9f9;
#         border-left: 5px solid #2c3e50;
#         padding: 1rem;
#         margin: 1rem 0;
#         border-radius: 4px;
#     }
#     .info-box h4 {
#         margin: 0;
#         color: #2c3e50;
#     }
#     .info-box p {
#         margin: 0.5rem 0 0 0;
#         font-size: 1.1rem;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# def create_info_box(title, value):
#     """
#     Returns HTML code for an info box with a title and value.
#     """
#     return f"""
#     <div class="info-box">
#         <h4>{title}</h4>
#         <p>{value}</p>
#     </div>
#     """

# # ------------------------------
# # Sidebar Inputs
# # ------------------------------
# st.sidebar.header("Input Parameters")
# S0 = st.sidebar.number_input("Spot Price (S0)", value=100.0, step=1.0)
# K = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0)
# T = st.sidebar.number_input("Time to Maturity (T in Years)", value=1.0, step=0.1)
# r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05, step=0.01)
# sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01)

# st.sidebar.header("Discretization Parameters")
# dS = st.sidebar.number_input("Stock Price Step (dS)", value=10.0, step=0.1)
# dt = st.sidebar.number_input("Time Step (dt)", value=0.001, step=0.001)

# st.sidebar.header("Option Settings")
# option_type = st.sidebar.selectbox("Option Type", ("Call", "Put"))
# numerical_method = st.sidebar.selectbox("Numerical Method",
#                                           ("Forward Euler", "Backward Euler", "Crank-Nicolson"))

# # ------------------------------
# # Run the "Simulation"
# # ------------------------------
# if st.button("Compute Option Prices"):
    
#     # Compute the numerical price and grid based on the selected method
#     if numerical_method == "Forward Euler":
#         num_price, S_grid, numeric_prices = forward_euler(S0, K, T, r, sigma, dS, dt, option_type)
#     elif numerical_method == "Backward Euler":
#         num_price, S_grid, numeric_prices = backward_euler(S0, K, r, T, sigma, dS, dt, option_type)
#     elif numerical_method == "Crank-Nicolson":
#         num_price, S_grid, numeric_prices = crank_nicolson(S0, K, r, T, sigma, dS, dt, option_type)
    
#     # Compute analytical Black–Scholes prices over the stock grid & at S0
#     analytical_prices = black_scholes(S_grid, K, T, r, sigma, option_type)
#     analytical_price = black_scholes(S0, K, T, r, sigma, option_type)
    
#     # Compute the absolute error at the spot price
#     abs_error = abs(num_price - analytical_price)
    
#     # ------------------------------
#     # Display Info Boxes
#     # ------------------------------
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.markdown(create_info_box("Numerical Price @ S0", f"${num_price:,.4f}"), unsafe_allow_html=True)
#     with col2:
#         st.markdown(create_info_box("Analytical Price @ S0", f"${analytical_price:,.4f}"), unsafe_allow_html=True)
#     with col3:
#         st.markdown(create_info_box("Absolute Error", f"${abs_error:,.4f}"), unsafe_allow_html=True)
    
#     # ------------------------------
#     # Display Comparison Table
#     # ------------------------------
#     df = pd.DataFrame({
#         f"{numerical_method} Price": [np.round(num_price, 4)],
#         "Analytical Price": [np.round(analytical_price, 4)],
#         "Absolute Error": [np.round(abs_error, 4)]
#     })
#     st.write("### Price Comparison at Spot Price (S0)")
#     st.table(df)
    
#     # ------------------------------
#     # Display Graphs in Two Columns
#     # ------------------------------
#     # Graph 1: Price Comparison Plot
#     col1, col2 = st.columns(2)
#     with col1:
#         fig_prices = go.Figure()
#         # Plot the numerical method prices as markers.
#         fig_prices.add_trace(go.Scatter(
#             x=S_grid,
#             y=numeric_prices,
#             mode='markers',
#             marker=dict(color="red", size=8),
#             name=f"{numerical_method} Prices"
#         ))
#         # Plot the analytical Black-Scholes prices as a line.
#         fig_prices.add_trace(go.Scatter(
#             x=S_grid,
#             y=analytical_prices,
#             mode='lines',
#             line=dict(color="blue", width=2),
#             name="Analytical Prices"
#         ))
#         fig_prices.update_layout(
#             title="Option Prices Across the Stock Price Grid",
#             xaxis_title="Stock Price (S)",
#             yaxis_title="Option Price (V)"
#         )
#         st.plotly_chart(fig_prices, use_container_width=True)
    
#     # Graph 2: Absolute Error Across the Stock Price Grid
#     with col2:
#         errors = np.abs(numeric_prices - analytical_prices)
#         fig_error = go.Figure()
#         fig_error.add_trace(go.Scatter(
#             x=S_grid,
#             y=errors,
#             mode='lines+markers',
#             line=dict(color="purple", width=2),
#             marker=dict(size=6),
#             name="Absolute Error"
#         ))
#         fig_error.update_layout(
#             title="Absolute Error Across the Stock Price Grid",
#             xaxis_title="Stock Price (S)",
#             yaxis_title="Absolute Error"
#         )
#         st.plotly_chart(fig_error, use_container_width=True)    








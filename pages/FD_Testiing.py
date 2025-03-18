import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import interp1d

###############################################################################
# Analytical Black-Scholes formula (vanilla only)
###############################################################################
def black_scholes(S, K, T, r, sigma, option_type):
    """Vanilla Black-Scholes formula for reference (call or put)."""
    # Avoid division by zero if T=0
    if T <= 1e-12:
        payoff = max(S - K, 0) if option_type == "Call" else max(K - S, 0)
        return payoff

    d1 = (np.log(S / K + 1e-99) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) + 1e-12)
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "Call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

###############################################################################
# PDE Solvers for Vanilla (No Barrier)
###############################################################################
def forward_euler_vanilla(S0, K, T, r, sigma, dS, dt, option_type):
    """
    Forward Euler PDE for a *vanilla* (no barrier) European call/put.
    Returns: (price_at_S0, S_grid, option_values_at_t0).
    """
    S_max = 2 * max(S0, K) * np.exp(r * T)
    M = int(round(S_max / dS))
    N = int(round(T / dt))
    # Recompute exact dS, dt so the grid lines up
    dS = S_max / M
    dt = T / N

    matval = np.zeros((M + 1, N + 1))
    vetS = np.linspace(0, S_max, M + 1)

    # Final condition (at expiry)
    if option_type == "Call":
        matval[:, -1] = np.maximum(vetS - K, 0)
    else:  # put
        matval[:, -1] = np.maximum(K - vetS, 0)

    # Boundary conditions (for each time slice)
    # left boundary (S=0)
    if option_type == "Call":
        matval[0, :] = 0.0
    else:  # put
        # put(0) ~ K*e^{-r*(T - t)}, we approximate with discrete steps
        t_vec = np.linspace(0, T, N + 1)
        matval[0, :] = K * np.exp(-r * (T - t_vec))

    # right boundary (S = S_max)
    if option_type == "Call":
        t_vec = np.linspace(0, T, N + 1)
        matval[-1, :] = (S_max - K * np.exp(-r * (T - t_vec)))
    else:  # put
        matval[-1, :] = 0.0

    # Coefficients for the PDE
    # i-values in [0..M]
    i_idx = np.arange(M + 1)
    a = 0.5 * dt * (sigma**2 * i_idx**2 - r * i_idx)
    b = 1 - dt * (sigma**2 * i_idx**2 + r)
    c = 0.5 * dt * (sigma**2 * i_idx**2 + r * i_idx)

    # Time stepping backward from j=N to j=0
    for j in range(N, 0, -1):
        for i in range(1, M):
            matval[i, j - 1] = a[i] * matval[i - 1, j] \
                               + b[i] * matval[i, j] \
                               + c[i] * matval[i + 1, j]

    # Interpolate the t=0 slice to find the price at S0
    price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
    price = price_interp(S0)
    return price, vetS, matval[:, 0]


def backward_euler_vanilla(S0, K, T, r, sigma, dS, dt, option_type):
    S_max = 2 * max(S0, K) * np.exp(r * T)
    M = int(round(S_max / dS))
    N = int(round(T / dt))
    dS = S_max / M
    dt = T / N

    matval = np.zeros((M + 1, N + 1))
    vetS = np.linspace(0, S_max, M + 1)
    # Expiry payoff
    if option_type == "Call":
        matval[:, -1] = np.maximum(vetS - K, 0)
    else:
        matval[:, -1] = np.maximum(K - vetS, 0)

    # Boundary conditions
    # S=0
    t_vec = np.linspace(0, T, N + 1)
    if option_type == "Call":
        matval[0, :] = 0.0
        matval[-1, :] = S_max - K * np.exp(-r * (T - t_vec))  # large S boundary
    else:  # put
        matval[0, :] = K * np.exp(-r * (T - t_vec))
        matval[-1, :] = 0.0

    # set up the tri-diagonal system
    i_idx = np.arange(M + 1)
    a = 0.5 * (r * dt * i_idx - sigma**2 * dt * i_idx**2)
    b = 1 + sigma**2 * dt * i_idx**2 + r * dt
    c = -0.5 * (r * dt * i_idx + sigma**2 * dt * i_idx**2)

    # Coeff matrix dimension = (M-1) x (M-1)
    # We'll build it once, then do an LU factor
    # For 1..M-1 in i_idx
    Adata = np.zeros((M - 1, M - 1))
    for i in range(1, M):
        # main diagonal
        Adata[i - 1, i - 1] = b[i]
        # sub-diagonal
        if i - 1 >= 1:
            Adata[i - 1, i - 2] = a[i]
        # super-diagonal
        if i <= M - 2:
            Adata[i - 1, i] = c[i]

    LU, piv = lu_factor(Adata)

    # Time-stepping
    for j in range(N - 1, -1, -1):
        # Right-hand side is matval[1:M, j+1], plus adjustments for boundaries
        rhs = matval[1:M, j + 1].copy()
        # Adjust for boundary terms:
        # For a[1], we add -a[1]*matval[0, j], for c[M-1], we add -c[M-1]*matval[M, j]
        rhs[0] -= a[1] * matval[0, j]
        rhs[-1] -= c[M - 1] * matval[M, j]

        # Solve
        sol = lu_solve((LU, piv), rhs)
        matval[1:M, j] = sol

    price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
    price = price_interp(S0)
    return price, vetS, matval[:, 0]


def crank_nicolson_vanilla(S0, K, T, r, sigma, dS, dt, option_type):
    S_max = 2 * max(S0, K) * np.exp(r * T)
    M = int(round(S_max / dS))
    N = int(round(T / dt))
    dS = S_max / M
    dt = T / N

    matval = np.zeros((M + 1, N + 1))
    vetS = np.linspace(0, S_max, M + 1)

    # Final payoff
    if option_type == "Call":
        matval[:, -1] = np.maximum(vetS - K, 0)
    else:
        matval[:, -1] = np.maximum(K - vetS, 0)

    # BCs
    t_vec = np.linspace(0, T, N + 1)
    if option_type == "Call":
        matval[0, :] = 0.0
        matval[-1, :] = S_max - K * np.exp(-r * (T - t_vec))
    else:
        matval[0, :] = K * np.exp(-r * (T - t_vec))
        matval[-1, :] = 0.0

    # Build tri-diagonal for CN
    # alpha, beta, gamma
    # CN: 0.5 * (explicit + implicit)
    i_idx = np.arange(M + 1)
    alpha = 0.25 * dt * (sigma**2 * i_idx**2 - r * i_idx)
    beta = -0.5 * dt * (sigma**2 * i_idx**2 + r)
    gamma = 0.25 * dt * (sigma**2 * i_idx**2 + r * i_idx)

    # M1 = (I - 0.5*A), M2 = (I + 0.5*A)
    # where A is the matrix from alpha, beta, gamma
    # We'll construct them for i=1..M-1
    # Diagonal dimension = (M-1)
    M1 = np.zeros((M - 1, M - 1))
    M2 = np.zeros((M - 1, M - 1))

    for i in range(1, M):
        # main diagonal
        M1[i - 1, i - 1] = 1 - beta[i]
        M2[i - 1, i - 1] = 1 + beta[i]
        # sub-diagonal
        if i - 1 >= 1:
            M1[i - 1, i - 2] = -alpha[i]
            M2[i - 1, i - 2] = alpha[i]
        # super-diagonal
        if i <= M - 2:
            M1[i - 1, i] = -gamma[i]
            M2[i - 1, i] = gamma[i]

    LU1, piv1 = lu_factor(M1)  # factor M1 once

    for j in range(N - 1, -1, -1):
        # We want M1 * U(., j) = M2 * U(., j+1) + boundary terms
        rhs = M2 @ matval[1:M, j + 1]

        # Adjust for boundaries in RHS:
        # The sub/super diagonal can multiply the boundary nodes:
        # sub-diagonal => alpha[i] * matval[0, j+1]
        # super-diagonal => gamma[i] * matval[M, j+1]
        # We'll do an explicit small fix if needed:
        # left boundary => i=1 => M2[0,0], M2[0,-1]? etc.

        # alpha[1]*matval[0, j+1] and gamma[M-1]*matval[M, j+1]
        # Actually, we do it more precisely:
        rhs[0] += alpha[1] * matval[0, j + 1]
        rhs[-1] += gamma[M - 1] * matval[M, j + 1]

        sol = lu_solve((LU1, piv1), rhs)
        matval[1:M, j] = sol

    price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
    price = price_interp(S0)
    return price, vetS, matval[:, 0]

###############################################################################
# Barrier Logic: Knock-Out PDE
###############################################################################
def forward_euler_knock_out(S0, K, T, r, sigma, dS, dt, option_type,
                            barrier, barrier_type):
    """
    Forward Euler PDE for knock-out barrier:
      - "up-and-out" => zero the grid for S >= barrier
      - "down-and-out" => zero the grid for S <= barrier
    """
    # We do almost the same logic as forward_euler_vanilla, but each time-step
    # we enforce that any node that has crossed the barrier is set to 0.
    # If barrier_type = "up-and-out": zero if S >= barrier
    # If barrier_type = "down-and-out": zero if S <= barrier
    S_max = 2 * max(S0, K, barrier) * np.exp(r * T)  # ensure barrier within grid
    M = int(round(S_max / dS))
    N = int(round(T / dt))
    dS = S_max / M
    dt = T / N

    matval = np.zeros((M + 1, N + 1))
    vetS = np.linspace(0, S_max, M + 1)

    # Final condition
    if option_type == "Call":
        matval[:, -1] = np.maximum(vetS - K, 0)
    else:
        matval[:, -1] = np.maximum(K - vetS, 0)

    # Boundaries
    t_vec = np.linspace(0, T, N + 1)
    if option_type == "Call":
        matval[0, :] = 0.0
        matval[-1, :] = S_max - K * np.exp(-r * (T - t_vec))
    else:
        matval[0, :] = K * np.exp(-r * (T - t_vec))
        matval[-1, :] = 0.0

    # PDE coefficients
    i_idx = np.arange(M + 1)
    a = 0.5 * dt * (sigma**2 * i_idx**2 - r * i_idx)
    b = 1 - dt * (sigma**2 * i_idx**2 + r)
    c = 0.5 * dt * (sigma**2 * i_idx**2 + r * i_idx)

    # Helper to apply knock-out
    def apply_knock_out(Ucol):
        if barrier_type == "up-and-out":
            # zero where S >= barrier
            Ucol[vetS >= barrier] = 0.0
        elif barrier_type == "down-and-out":
            # zero where S <= barrier
            Ucol[vetS <= barrier] = 0.0

    # Apply knock-out at expiry
    apply_knock_out(matval[:, -1])

    # Time-stepping
    for j in range(N, 0, -1):
        for i in range(1, M):
            matval[i, j - 1] = a[i] * matval[i - 1, j] \
                               + b[i] * matval[i, j] \
                               + c[i] * matval[i + 1, j]
        # After computing the new column j-1, apply knock-out
        apply_knock_out(matval[:, j - 1])

    # Price at S0
    price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
    price = price_interp(S0)
    return price, vetS, matval[:, 0]


# (Similarly, you’d define backward_euler_knock_out(...) and crank_nicolson_knock_out(...).
#  For brevity, we'll show just the forward euler version in detail.
#  Below, we implement the others more concisely.)
def backward_euler_knock_out(S0, K, T, r, sigma, dS, dt, option_type,
                             barrier, barrier_type):
    """
    Backward Euler PDE for knock-out barrier.
    We'll reuse backward_euler_vanilla, but after each time step we zero out
    the knocked-out region.
    """
    # Solve vanilla PDE first
    price_van, S_grid, col_t0 = backward_euler_vanilla(S0, K, T, r, sigma, dS, dt, option_type)

    # We actually need the full matrix of values at each time step to apply knockout after each step.
    # But for simplicity, let's re-implement similarly to forward_euler_knock_out.
    # This code shows a minimal approach: we'll do the same steps as vanilla but apply
    # barrier logic after each time step. Implementation approach is analogous.
    # Due to code length, let's do a simpler approach:
    #  1) We'll do the same setup
    #  2) We'll do a time loop
    #  3) after each solve, apply knockout
    # (See forward Euler above for a detailed explanation.)
    
    S_max = 2 * max(S0, K, barrier) * np.exp(r * T)
    M = int(round(S_max / dS))
    N = int(round(T / dt))
    dS = S_max / M
    dt = T / N
    vetS = np.linspace(0, S_max, M + 1)

    matval = np.zeros((M + 1, N + 1))
    # expiry payoff
    if option_type == "Call":
        matval[:, -1] = np.maximum(vetS - K, 0)
    else:
        matval[:, -1] = np.maximum(K - vetS, 0)

    # boundaries
    t_vec = np.linspace(0, T, N + 1)
    if option_type == "Call":
        matval[0, :] = 0.0
        matval[-1, :] = S_max - K * np.exp(-r * (T - t_vec))
    else:
        matval[0, :] = K * np.exp(-r * (T - t_vec))
        matval[-1, :] = 0.0

    # knockout helper
    def apply_knock_out(Ucol):
        if barrier_type == "up-and-out":
            Ucol[vetS >= barrier] = 0.0
        else:  # down-and-out
            Ucol[vetS <= barrier] = 0.0

    # apply at expiry
    apply_knock_out(matval[:, -1])

    # tri-diagonal
    i_idx = np.arange(M + 1)
    a = 0.5 * (r * dt * i_idx - sigma**2 * dt * i_idx**2)
    b = 1 + sigma**2 * dt * i_idx**2 + r * dt
    c = -0.5 * (r * dt * i_idx + sigma**2 * dt * i_idx**2)

    Adata = np.zeros((M - 1, M - 1))
    for i in range(1, M):
        Adata[i - 1, i - 1] = b[i]
        if i - 1 >= 1:
            Adata[i - 1, i - 2] = a[i]
        if i <= M - 2:
            Adata[i - 1, i] = c[i]
    LU, piv = lu_factor(Adata)

    # time stepping
    for j in range(N - 1, -1, -1):
        rhs = matval[1:M, j + 1].copy()
        rhs[0] -= a[1] * matval[0, j]
        rhs[-1] -= c[M - 1] * matval[M, j]

        sol = lu_solve((LU, piv), rhs)
        matval[1:M, j] = sol

        # apply knockout to new column j
        apply_knock_out(matval[:, j])

    price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
    price = price_interp(S0)
    return price, vetS, matval[:, 0]


def crank_nicolson_knock_out(S0, K, T, r, sigma, dS, dt, option_type,
                             barrier, barrier_type):
    """
    Crank-Nicolson PDE for knock-out barrier.
    Similar approach: solve each step, then zero out knocked-out region.
    """
    # We'll copy the logic from crank_nicolson_vanilla, then apply knockout after each time step.
    S_max = 2 * max(S0, K, barrier) * np.exp(r * T)
    M = int(round(S_max / dS))
    N = int(round(T / dt))
    dS = S_max / M
    dt = T / N
    matval = np.zeros((M + 1, N + 1))
    vetS = np.linspace(0, S_max, M + 1)

    # payoff
    if option_type == "Call":
        matval[:, -1] = np.maximum(vetS - K, 0)
    else:
        matval[:, -1] = np.maximum(K - vetS, 0)

    t_vec = np.linspace(0, T, N + 1)
    if option_type == "Call":
        matval[0, :] = 0.0
        matval[-1, :] = S_max - K * np.exp(-r * (T - t_vec))
    else:
        matval[0, :] = K * np.exp(-r * (T - t_vec))
        matval[-1, :] = 0.0

    def apply_knock_out(Ucol):
        if barrier_type == "up-and-out":
            Ucol[vetS >= barrier] = 0.0
        else:
            Ucol[vetS <= barrier] = 0.0

    apply_knock_out(matval[:, -1])

    i_idx = np.arange(M + 1)
    alpha = 0.25 * dt * (sigma**2 * i_idx**2 - r * i_idx)
    beta = -0.5 * dt * (sigma**2 * i_idx**2 + r)
    gamma = 0.25 * dt * (sigma**2 * i_idx**2 + r * i_idx)

    # M1, M2
    M1 = np.zeros((M - 1, M - 1))
    M2 = np.zeros((M - 1, M - 1))

    for i in range(1, M):
        M1[i - 1, i - 1] = 1 - beta[i]
        M2[i - 1, i - 1] = 1 + beta[i]
        if i - 1 >= 1:
            M1[i - 1, i - 2] = -alpha[i]
            M2[i - 1, i - 2] = alpha[i]
        if i <= M - 2:
            M1[i - 1, i] = -gamma[i]
            M2[i - 1, i] = gamma[i]
    LU1, piv1 = lu_factor(M1)

    for j in range(N - 1, -1, -1):
        rhs = M2 @ matval[1:M, j + 1]
        # boundary adjustments
        rhs[0] += alpha[1] * matval[0, j + 1]
        rhs[-1] += gamma[M - 1] * matval[M, j + 1]

        sol = lu_solve((LU1, piv1), rhs)
        matval[1:M, j] = sol

        apply_knock_out(matval[:, j])

    price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
    price = price_interp(S0)
    return price, vetS, matval[:, 0]


###############################################################################
# Barrier Pricing Wrapper
###############################################################################
def barrier_price(method, S0, K, T, r, sigma, dS, dt, option_type,
                  barrier, barrier_kind):
    """
    barrier_kind in { 'None', 'Up-and-out', 'Down-and-out', 'Up-and-in', 'Down-and-in' }
    This function calls the appropriate PDE solver.
    For 'None' => vanilla PDE
    For 'Knock-out' => PDE with zeroing out
    For 'Knock-in' => vanilla PDE - knock-out PDE
    Returns (price_at_S0, S_grid, values_at_S_grid).
    """
    # 1) If there's no barrier:
    if barrier_kind == "None":
        if method == "Forward Euler":
            return forward_euler_vanilla(S0, K, T, r, sigma, dS, dt, option_type)
        elif method == "Backward Euler":
            return backward_euler_vanilla(S0, K, T, r, sigma, dS, dt, option_type)
        else:
            return crank_nicolson_vanilla(S0, K, T, r, sigma, dS, dt, option_type)

    # 2) If it's up-and-out or down-and-out, do knock-out PDE:
    if barrier_kind == "Up-and-out" or barrier_kind == "Down-and-out":
        if method == "Forward Euler":
            return forward_euler_knock_out(S0, K, T, r, sigma, dS, dt, option_type,
                                           barrier, barrier_kind.split('-')[0].lower())
        elif method == "Backward Euler":
            return backward_euler_knock_out(S0, K, T, r, sigma, dS, dt, option_type,
                                            barrier, barrier_kind.split('-')[0].lower())
        else:
            return crank_nicolson_knock_out(S0, K, T, r, sigma, dS, dt, option_type,
                                            barrier, barrier_kind.split('-')[0].lower())

    # 3) If it's up-and-in or down-and-in => knock-in = vanilla - knock-out
    if barrier_kind == "Up-and-in" or barrier_kind == "Down-and-in":
        # first do vanilla
        if method == "Forward Euler":
            vanilla_price, S_grid, vanilla_slice = forward_euler_vanilla(
                S0, K, T, r, sigma, dS, dt, option_type
            )
            ko_price, _, ko_slice = forward_euler_knock_out(
                S0, K, T, r, sigma, dS, dt, option_type, barrier,
                barrier_kind.split('-')[0].lower()
            )
        elif method == "Backward Euler":
            vanilla_price, S_grid, vanilla_slice = backward_euler_vanilla(
                S0, K, T, r, sigma, dS, dt, option_type
            )
            ko_price, _, ko_slice = backward_euler_knock_out(
                S0, K, T, r, sigma, dS, dt, option_type, barrier,
                barrier_kind.split('-')[0].lower()
            )
        else:
            vanilla_price, S_grid, vanilla_slice = crank_nicolson_vanilla(
                S0, K, T, r, sigma, dS, dt, option_type
            )
            ko_price, _, ko_slice = crank_nicolson_knock_out(
                S0, K, T, r, sigma, dS, dt, option_type, barrier,
                barrier_kind.split('-')[0].lower()
            )
        knock_in_price = vanilla_price - ko_price
        knock_in_slice = vanilla_slice - ko_slice
        return knock_in_price, S_grid, knock_in_slice

    # Default fallback (should not happen):
    return 0.0, np.array([]), np.array([])

###############################################################################
# Streamlit UI
###############################################################################
st.title("Finite-Difference Pricing of Barrier Options (and Vanilla)")

S0 = st.sidebar.number_input("Spot Price (S0)", value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0, step=0.1)
r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05, step=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01)
dS = st.sidebar.number_input("Stock Price Step (dS)", value=5.0, step=1.0)
dt = st.sidebar.number_input("Time Step (dt)", value=0.01, step=0.01)
option_type = st.sidebar.selectbox("Option Type", ("Call", "Put"))

# New: Barrier
barrier_kind = st.sidebar.selectbox(
    "Barrier Type",
    ["None", "Up-and-out", "Down-and-out", "Up-and-in", "Down-and-in"]
)
barrier_level = 0.0
if barrier_kind != "None":
    barrier_level = st.sidebar.number_input("Barrier Level", value=120.0, step=1.0)

numerical_method = st.sidebar.selectbox(
    "Numerical method",
    ("Forward Euler", "Backward Euler", "Crank-Nicolson")
)

# Price the chosen barrier (or vanilla) with the selected FD method
price, S_grid, fd_prices = barrier_price(
    numerical_method, S0, K, T, r, sigma, dS, dt,
    option_type, barrier_level, barrier_kind
)

# We'll also show the standard Black-Scholes curve (which is for a *vanilla* option),
# just for reference. Note that for barrier options, there's no direct match to this
# vanilla line, but it's illustrative to see the difference.
vanilla_bs_prices = np.array([black_scholes(s, K, T, r, sigma, option_type) for s in S_grid])
bs_price_at_S0 = black_scholes(S0, K, T, r, sigma, option_type)

# Show a table comparing PDE vs. vanilla Black-Scholes (the latter is only correct for "None")
df = pd.DataFrame({
    "FD Price (at S0)": [np.round(price, 4)],
    "Vanilla B-S (at S0)": [bs_price_at_S0],
    "Absolute Error vs. Vanilla?": [abs(price - bs_price_at_S0)]
})
st.subheader(f"Computed Price at S0 ({barrier_kind})")
st.table(df)

# Plot
st.subheader("Comparison of FD Price Curve vs. Vanilla Black-Scholes")
fig = go.Figure()
# FD prices
fig.add_trace(go.Scatter(
    x=S_grid, 
    y=fd_prices, 
    mode="markers", 
    name=f"FD {barrier_kind}",
    marker=dict(color="red", size=5)
))
# Vanilla BS
fig.add_trace(go.Scatter(
    x=S_grid, 
    y=vanilla_bs_prices, 
    mode="lines", 
    name="Vanilla Black-Scholes",
    line=dict(color="blue", width=2)
))
fig.update_layout(
    title=f"PDE vs. Vanilla B-S (Barrier: {barrier_kind})",
    xaxis_title="Stock Price (S)",
    yaxis_title="Option Price",
    legend_title="Method",
    width=800,
    height=500
)
st.plotly_chart(fig)

st.write("""
**Note**:  
- If "Barrier Type" is "None", the PDE line should match the vanilla Black–Scholes line more closely for sufficiently small dS and dt.  
- For actual barrier options ("Knock-Out" or "Knock-In"), there's no direct Black–Scholes line in this code, so we plot the vanilla curve for reference only.  
- "Knock-In = Vanilla − Knock-Out" is a standard replication approach.  
- You can refine the numerical accuracy by decreasing dt and dS, but it may slow down the computation.
""")

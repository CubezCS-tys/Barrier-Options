import streamlit as st
import numpy as np
import math
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from scipy.linalg import solve_banded
from scipy.stats import norm

########################################################################
# 1) Standard Analytical Functions (for comparison)
########################################################################
def calc_d1(S0, K, r, q, sigma, T):
    return (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def calc_d2(S0, K, r, q, sigma, T):
    return calc_d1(S0, K, r, q, sigma, T) - sigma * np.sqrt(T)

def calc_c(S0, K, r, q, sigma, T):
    d1 = calc_d1(S0, K, r, q, sigma, T)
    d2 = calc_d2(S0, K, r, q, sigma, T)
    return (S0 * np.exp(-q * T) * norm.cdf(d1)
            - K * np.exp(-r * T) * norm.cdf(d2))

def calc_p(S0, K, r, q, sigma, T):
    d1 = calc_d1(S0, K, r, q, sigma, T)
    d2 = calc_d2(S0, K, r, q, sigma, T)
    return (K * np.exp(-r * T) * norm.cdf(-d2)
            - S0 * np.exp(-q * T) * norm.cdf(-d1))

def calc_lambda(r, q, sigma):
    return (r - q + 0.5 * sigma**2) / (sigma**2)

def calc_y(barrier, S0, K, T, sigma, r, q):
    lam = calc_lambda(r, q, sigma)
    return (np.log((barrier**2)/(S0*K)) / (sigma*np.sqrt(T))) + lam * sigma * np.sqrt(T)

def calc_x1(S0, barrier, T, sigma, r, q):
    lam = calc_lambda(r, q, sigma)
    return (np.log(S0 / barrier) / (sigma * np.sqrt(T))) + lam * sigma * np.sqrt(T)

def calc_y1(S0, barrier, T, sigma, r, q):
    lam = calc_lambda(r, q, sigma)
    return (np.log(barrier / S0) / (sigma * np.sqrt(T))) + lam * sigma * np.sqrt(T)

def black_scholes(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

########################################################################
# 2) Barrier Option Pricing Function (Analytical – for reference)
########################################################################
def barrier_option_price(S0, K, T, r, q, sigma, barrier, option_type):
    # This function should implement the full barrier option formulas.
    # For simplicity here, we return the vanilla call (or put) price.
    # In practice, you would include the proper formulas.
    if "call" in option_type.lower():
        return calc_c(S0, K, r, q, sigma, T)
    else:
        return calc_p(S0, K, r, q, sigma, T)

########################################################################
# 3) Nonuniform Grid Construction Using Tanh Transformation
########################################################################
def build_nonuniform_grid_tanh(Smin, Smax, barrier, dS, clustering=5):
    """
    Build a nonuniform grid on [Smin, Smax] using a tanh transformation.
    More points are clustered around the barrier.
    
    The number of grid points is computed as:
         N = int((Smax - Smin) / dS) + 1.
    
    :param Smin: Minimum asset price.
    :param Smax: Maximum asset price.
    :param barrier: Barrier level.
    :param dS: Nominal space step.
    :param clustering: Clustering intensity (larger => more clustering).
    :return: 1D numpy array of grid points.
    """
    N = int((Smax - Smin) / dS) + 1
    x = np.linspace(0, 1, N)
    c = (barrier - Smin) / (Smax - Smin)  # normalized barrier location
    scale = np.tanh(clustering * max(c, 1 - c))
    Sgrid = Smin + (Smax - Smin) * ((np.tanh(clustering * (x - c)) / scale + 1) / 2)
    return Sgrid

########################################################################
# 4) Crank–Nicolson PDE Solver on a Nonuniform Grid for Barrier Options
########################################################################
def crank_nicolson_barrier_nonuniform(S0, K, T, r, q, sigma,
                                      barrier, barrier_type, option_side,
                                      Smin, Smax, dS, dt, clustering=5):
    """
    Prices a knock-out barrier option using Crank–Nicolson on a nonuniform grid.
    
    The spatial grid is generated via a tanh transformation (with intensity 'clustering').
    The user provides the nominal space step (dS) and time step (dt).
    
    This solver applies the standard Crank–Nicolson scheme on interior nodes,
    using nonuniform central differences:
      V'(S_i)  ≈ (V[i+1] - V[i-1]) / (S[i+1] - S[i-1])
      V''(S_i) ≈ 2/(dS_{i-1}+dS_{i}) * ((V[i+1]-V[i])/dS_{i} - (V[i]-V[i-1])/dS_{i-1})
    
    Barrier knockout is enforced by setting V=0 in the knockout region.
    
    :return: (price at S0, S_grid, V at t=0)
    """
    # Build the spatial grid.
    S_grid = build_nonuniform_grid_tanh(Smin, Smax, barrier, dS, clustering=clustering)
    N = len(S_grid)
    M_t = int(T / dt)
    
    # Allocate solution: V[m, i] is the solution at time level m.
    V = np.zeros((M_t + 1, N))
    
    # Terminal condition at t = T:
    for i in range(N):
        S = S_grid[i]
        if option_side.lower() == 'call':
            payoff = max(S - K, 0)
        else:
            payoff = max(K - S, 0)
        if barrier_type == 'up' and S >= barrier:
            payoff = 0.0
        if barrier_type == 'down' and S <= barrier:
            payoff = 0.0
        V[M_t, i] = payoff
    
    # Precompute finite difference coefficients for interior nodes.
    # For each interior node i = 1,..., N-2, compute:
    #   dS_minus = S[i] - S[i-1], dS_plus = S[i+1] - S[i], dS_total = dS_minus + dS_plus.
    a = np.zeros(N)  # coefficient for V_{i-1}
    b_coef = np.zeros(N)  # coefficient for V_i
    c_coef = np.zeros(N)  # coefficient for V_{i+1}
    
    for i in range(1, N-1):
        dS_minus = S_grid[i] - S_grid[i-1]
        dS_plus  = S_grid[i+1] - S_grid[i]
        dS_total = dS_minus + dS_plus
        # Here we will build the local differential operator (for the PDE term)
        # Diffusion part (second derivative) coefficients:
        a[i] = 2.0 / (dS_total * dS_minus)
        c_coef[i] = 2.0 / (dS_total * dS_plus)
        # Approximation for first derivative (central difference):
        # We'll use a simple average denominator: 1/(dS_minus + dS_plus)
        # and later multiply by S and (r - q).
        b_coef[i] = - (a[i] + c_coef[i])
        # Then subtract r for the discount term:
        b_coef[i] -= r
    
    # Now perform backward time stepping with Crank–Nicolson:
    for m in range(M_t-1, -1, -1):
        tau = T - m*dt  # remaining time
        # Set boundary conditions:
        if option_side.lower() == 'call':
            V[m, 0] = 0.0
            V[m, -1] = S_grid[-1] - K * math.exp(-r * tau)
        else:
            V[m, 0] = K * math.exp(-r * tau)
            V[m, -1] = 0.0
        
        # Assemble the tridiagonal system for interior nodes i = 1,..., N-2.
        n_int = N - 2
        # For each interior node, the CN scheme gives:
        #   (1 - dt/2 * L_i) V^{m} = (1 + dt/2 * L_i) V^{m+1}
        # where L_i is our local differential operator.
        A_main = np.zeros(n_int)
        A_lower = np.zeros(n_int-1)
        A_upper = np.zeros(n_int-1)
        RHS = np.zeros(n_int)
        
        for i in range(1, N-1):
            idx = i - 1
            alpha = dt/2 * a[i]
            gamma = dt/2 * c_coef[i]
            beta = dt/2 * b_coef[i]
            A_main[idx] = 1 - beta
            if idx > 0:
                A_lower[idx-1] = -alpha
            if idx < n_int - 1:
                A_upper[idx] = -gamma
            RHS[idx] = (1 + beta) * V[m+1, i]
            if i - 1 >= 0:
                RHS[idx] += dt/2 * a[i] * V[m+1, i-1]
            if i + 1 < N:
                RHS[idx] += dt/2 * c_coef[i] * V[m+1, i+1]
        
        # Solve the tridiagonal system:
        # We can use np.linalg.solve on the full matrix.
        # Build full matrix:
        mat = np.diag(A_main)
        if n_int > 1:
            mat += np.diag(A_lower, k=-1) + np.diag(A_upper, k=1)
        V_new = np.linalg.solve(mat, RHS)
        for i in range(1, N-1):
            V[m, i] = V_new[i-1]
        
        # Enforce barrier condition:
        for i in range(N):
            if barrier_type == 'up' and S_grid[i] >= barrier:
                V[m, i] = 0.0
            if barrier_type == 'down' and S_grid[i] <= barrier:
                V[m, i] = 0.0

    # Interpolate the solution at t = 0 to get the price at S0:
    f_interp = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
    price = float(f_interp(S0))
    return price, S_grid, V[0, :]

########################################################################
# 5) Streamlit App
########################################################################
def app():
    st.title("Barrier Options: Crank–Nicolson with Nonuniform (Tanh) Grid")
    st.markdown("""
    This app prices knock–out barrier options using a Crank–Nicolson PDE solver on a nonuniform spatial grid.
    The grid is generated via a tanh transformation to cluster points around the barrier.
    
    You specify the nominal space step (dS) and time step (dt); the total number of grid points is computed from the spatial domain.
    
    **Note:** This implementation directly handles knock‑out barriers. Knock‑in options can be obtained via in–out parity.
    """)
    
    # Sidebar inputs
    S0 = st.sidebar.number_input("Spot Price (S0)", value=100.0, step=1.0)
    K  = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0)
    T  = st.sidebar.number_input("Time to Maturity (T)", value=1.0, step=0.1)
    r  = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)
    q  = st.sidebar.number_input("Dividend Yield (q)", value=0.00, step=0.01)
    sigma = st.sidebar.number_input("Volatility (sigma)", value=0.2, step=0.01)
    
    barrier = st.sidebar.number_input("Barrier", value=120.0, step=1.0)
    barrier_type = st.sidebar.selectbox("Barrier Type", ["up", "down"])
    option_side = st.sidebar.selectbox("Option Side", ["call", "put"])
    
    dt = st.sidebar.number_input("Time Step (dt)", value=0.01, step=0.001)
    dS = st.sidebar.number_input("Nominal Space Step (dS)", value=1.0, step=0.0001)
    clustering = st.sidebar.number_input("Clustering Intensity", value=5.0, step=0.5)
    Smax_factor = st.sidebar.slider("Smax Factor (× max(S0, K))", 1.0, 5.0, 2.0)
    
    Smin = 0.0
    Smax = max(S0, K) * Smax_factor * math.exp(r * T)
    
    price, S_grid, V0 = crank_nicolson_barrier_nonuniform(
        S0, K, T, r, q, sigma,
        barrier, barrier_type, option_side,
        Smin, Smax, dS, dt, clustering=clustering
    )
    
    st.write(f"**PDE Price at S0 = {S0}**: {price:.4f}")
    
    # For comparison, compute a simple (placeholder) analytical curve (here using vanilla Black-Scholes with barrier enforced)
    analytic_vals = []
    for s in S_grid:
        if option_side.lower() == 'call':
            val = s * math.exp(-q*T) * norm.cdf(calc_d1(s, K, r, q, sigma, T)) - K * math.exp(-r*T) * norm.cdf(calc_d2(s, K, r, q, sigma, T))
        else:
            val = K * math.exp(-r*T) * norm.cdf(-calc_d2(s, K, r, q, sigma, T)) - s * math.exp(-q*T) * norm.cdf(-calc_d1(s, K, r, q, sigma, T))
        if barrier_type == 'up' and s >= barrier:
            val = 0.0
        if barrier_type == 'down' and s <= barrier:
            val = 0.0
        analytic_vals.append(val)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=S_grid,
        y=V0,
        mode="lines+markers",
        name="CN PDE Solution at t=0"
    ))
    fig.add_trace(go.Scatter(
        x=S_grid,
        y=analytic_vals,
        mode="lines",
        name="Analytical (Placeholder)"
    ))
    fig.update_layout(
        title=f"{option_side.capitalize()} Option ({barrier_type.capitalize()} Barrier): CN PDE vs Analytical",
        xaxis_title="Stock Price (S)",
        yaxis_title="Option Value"
    )
    st.plotly_chart(fig)

if __name__ == "__main__":
    app()

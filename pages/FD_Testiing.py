import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Utility functions
###############################################################################

def vanilla_payoff(S, K, option_type="call"):
    """
    Computes the payoff of a European call/put at maturity.
    :param S: Spot array (or scalar).
    :param K: Strike price.
    :param option_type: "call" or "put"
    :return: Payoff array (or scalar).
    """
    if option_type == "call":
        return np.maximum(S - K, 0.0)
    else:
        return np.maximum(K - S, 0.0)

def apply_knock_out_condition(U, S_grid, barrier, barrier_type):
    """
    Zeroes out values in U that have 'knocked out'.
    :param U: The option value array, shape = (Nspace,).
    :param S_grid: The array of underlying prices corresponding to U.
    :param barrier: The barrier level.
    :param barrier_type: "up" or "down".
    :return: modified U (in-place).
    """
    if barrier_type == "up":
        # If S >= barrier, the option knocks out
        U[S_grid >= barrier] = 0.0
    else: # "down"
        # If S <= barrier, the option knocks out
        U[S_grid <= barrier] = 0.0

def price_knock_out_vanilla(method, S_grid, r, sigma, K, T, M, N,
                            barrier, barrier_type, option_type):
    """
    Price a knock-out (up-and-out or down-and-out) barrier option
    by directly applying zero boundary at knocked-out region
    in a PDE solver. Simplistic approach: we re-zero after each time step.
    """
    # Grid spacing
    ds = S_grid[1] - S_grid[0]
    dt = T / N
    # Coefficients for PDE
    # in the PDE: dU/dt + r S dU/dS + 0.5 sigma^2 S^2 d2U/dS^2 - rU = 0
    # We'll define a, b, c arrays for standard FD:
    #   alpha_i = 0.5 * dt * (sigma^2 * i^2 - r * i)
    #   beta_i  = 1 - dt * (sigma^2 * i^2 + r)
    #   gamma_i = 0.5 * dt * (sigma^2 * i^2 + r * i)
    
    # For i in [1..M-1], where i indexes in space
    # S_i = i * ds (assuming S_grid[0] = 0 if scaled that way, but we have a general grid)
    
    # We'll store the option values in U
    U = vanilla_payoff(S_grid, K, option_type)
    
    # For convenience, define some arrays for the coefficients:
    # But we must convert from S to "i" carefully. We'll do it in a loop.

    # Prepare tri-diagonal coefficients for the implicit part (if used)
    # We'll store them so that forward/backward/Crank can share them.
    i_values = np.arange(len(S_grid))
    # Because S_grid might not be i*ds exactly, we approximate i as S/ds for the formula
    i_float = S_grid / ds
    alpha = 0.5 * dt * (sigma**2 * i_float**2 - r * i_float)
    beta  = 1.0 - dt * (sigma**2 * i_float**2 + r)
    gamma = 0.5 * dt * (sigma**2 * i_float**2 + r * i_float)

    def apply_boundary_conditions(U_t):
        # At S=0 (left boundary), for calls payoff = 0, for puts payoff = K*e^{-r*(T-t)}
        # We'll apply a "linear" boundary or Dirichlet depending on option type:
        # This is simplistic: call(0) ~ 0, put(0) ~ K * e^{-r tau}, but we do a direct approach:
        if option_type == "call":
            U_t[0] = 0.0
        else:  # put
            U_t[0] = K * np.exp(-r*(0.0))  # might approximate as we move in time
        # At S max, for calls: ~ S-K e^{-r*(T-t)}, for puts ~ 0
        # We'll do a simplistic approach:
        if option_type == "call":
            U_t[-1] = S_grid[-1] - K * np.exp(-r*(0.0))
            U_t[-1] = max(U_t[-1], 0.0)
        else:
            U_t[-1] = 0.0

    # We define separate routines for forward, backward, and CN steps:

    def forward_euler_step(U_t):
        # U_{t+dt}(i) = U_t(i) + alpha_i * U_t(i-1) + beta_i * U_t(i) + gamma_i * U_t(i+1)
        U_next = np.zeros_like(U_t)
        for i in range(1, len(S_grid)-1):
            U_next[i] = (U_t[i]
                         + alpha[i] * U_t[i-1]
                         + beta[i]  * U_t[i]
                         + gamma[i] * U_t[i+1])
        # boundary
        apply_boundary_conditions(U_next)
        # barrier knock-out
        apply_knock_out_condition(U_next, S_grid, barrier, barrier_type)
        return U_next

    # For backward Euler and CN, we need to solve a linear system A * U_{t+1} = B * U_t
    # backward Euler => A = I - M   and B = I
    # M is the matrix with tri-diagonal from alpha, beta, gamma.
    # We'll construct the matrix once.

    # For i from 1 to M-1:
    #   -alpha_i * U(i-1) + (1 - beta_i) * U(i) - gamma_i * U(i+1) = U_t(i)
    # But note sign differences; carefully define the tri-diagonal.

    # Build tri-diagonal for backward Euler and CN
    # Let’s define them in a standard form:
    #  diagL[i] = -alpha[i], diagM[i] = (1 - beta[i]), diagU[i] = -gamma[i]
    # Then for CN we combine half step explicit + half step implicit
    diagL = np.zeros(len(S_grid))
    diagM = np.zeros(len(S_grid))
    diagU = np.zeros(len(S_grid))

    for i in range(1, len(S_grid)-1):
        diagL[i] = -alpha[i]
        diagM[i] = 1.0 + alpha[i] + gamma[i]
        diagU[i] = -gamma[i]

    def solve_tridiagonal(L, M, U_, rhs):
        """
        Solve a tri-diagonal system A x = rhs
        where A has sub-diagonal L, diagonal M, super-diagonal U_.
        Using simple Thomas algorithm.
        """
        n = len(rhs)
        # Forward pass
        for i in range(1,n):
            w = L[i]/M[i-1]
            M[i] = M[i] - w*U_[i-1]
            rhs[i] = rhs[i] - w*rhs[i-1]
        # Back substitution
        x = np.zeros(n)
        x[-1] = rhs[-1]/M[-1]
        for i in reversed(range(n-1)):
            x[i] = (rhs[i] - U_[i]*x[i+1]) / M[i]
        return x

    def backward_euler_step(U_t):
        # A = I - M => in components:
        # for i from 1..M-1:
        #   U_{t+dt}(i) - alpha_i U_{t+dt}(i-1) - gamma_i U_{t+dt}(i+1)
        #   = U_t(i) - beta_i U_{t+dt}(i)
        # We prepared diagL, diagM, diagU for the LHS = (1 + alpha+gamma) on diagonal, etc.
        # But we must be careful with boundary and forcing terms.
        # We'll create a copy of the tri-diagonal so as not to overwrite it each step.
        b = U_t.copy()
        # apply boundary in b:
        apply_boundary_conditions(b)
        # solve the system
        U_next = solve_tridiagonal(diagL.copy(), diagM.copy(), diagU.copy(), b)
        # boundary
        apply_boundary_conditions(U_next)
        # barrier knock-out
        apply_knock_out_condition(U_next, S_grid, barrier, barrier_type)
        return U_next

    def crank_nicolson_step(U_t):
        # half explicit + half implicit
        # CN: U_{t+dt} - 0.5*M U_{t+dt} = U_t + 0.5*M U_t
        # We'll do:
        #   LHS: (I - 0.5 M)
        #   RHS: (I + 0.5 M) U_t
        # Construct M explicitly from alpha, beta, gamma. Then do the factor for half dt.
        # We'll handle it directly with the tri-diagonal approach:
        # We'll define an "A" for LHS and "B" for RHS.
        
        # Step 1: B * U_t
        U_star = np.zeros_like(U_t)
        for i in range(1, len(S_grid)-1):
            U_star[i] = (U_t[i]
                         + 0.5 * (alpha[i]*U_t[i-1] + beta[i]*U_t[i] + gamma[i]*U_t[i+1]))
        
        # We'll handle boundaries in U_star
        apply_boundary_conditions(U_star)

        # Now solve (I - 0.5M) U_{t+dt} = U_star
        # We'll define tri-diagonal for A = I - 0.5*[ tri-diag from alpha, beta, gamma ]
        # That is:
        #   A diag : 1 - 0.5*beta[i]
        #   A lower: -0.5*alpha[i]
        #   A upper: -0.5*gamma[i]
        # We already have diagL, diagM, diagU for the backward approach. Let's build new arrays:
        L_A = np.zeros_like(diagL)
        M_A = np.zeros_like(diagM)
        U_A = np.zeros_like(diagU)
        for i in range(1, len(S_grid)-1):
            L_A[i] = -0.5 * alpha[i]
            M_A[i] = 1.0 + 0.5*(alpha[i] + gamma[i])
            U_A[i] = -0.5 * gamma[i]
        
        # Solve
        U_next = solve_tridiagonal(L_A, M_A, U_A, U_star)
        # apply boundary conditions
        apply_boundary_conditions(U_next)
        # barrier knock-out
        apply_knock_out_condition(U_next, S_grid, barrier, barrier_type)
        return U_next

    # Time stepping
    if method == "forward_euler":
        for _ in range(N):
            U = forward_euler_step(U)
    elif method == "backward_euler":
        for _ in range(N):
            U = backward_euler_step(U)
    elif method == "crank_nicolson":
        for _ in range(N):
            U = crank_nicolson_step(U)
    else:
        raise ValueError("Unknown method")

    return U

def price_knock_in_vanilla(method, S_grid, r, sigma, K, T, M, N,
                           barrier, barrier_type, option_type):
    """
    Price a knock-in barrier option using the relationship:
    Price(Knock-In) = Price(vanilla) - Price(Knock-Out).
    This is a commonly used static replication approach.
    """
    # Price vanilla European with PDE (no barrier):
    U_vanilla = price_knock_out_vanilla(method, S_grid, r, sigma, K, T, M, N,
                                        barrier=np.inf if barrier_type=="up" else -np.inf,
                                        barrier_type=barrier_type,
                                        option_type=option_type)
    # Price knock-out with the actual barrier
    U_knock_out = price_knock_out_vanilla(method, S_grid, r, sigma, K, T, M, N,
                                          barrier=barrier,
                                          barrier_type=barrier_type,
                                          option_type=option_type)
    return U_vanilla - U_knock_out

###############################################################################
# Streamlit App
###############################################################################

def main():
    st.title("Barrier Options PDE Pricing (Forward Euler, Backward Euler, Crank–Nicolson)")
    st.write("""
    This app prices the eight standard barrier options (Call/Put, Up/Down, In/Out)
    using finite-difference methods. 
    """)

    # Sidebar for user inputs
    st.sidebar.header("Model Parameters")
    S0 = st.sidebar.number_input("Spot price (S0)", value=100.0, min_value=0.01, step=1.0)
    K = st.sidebar.number_input("Strike (K)", value=100.0, min_value=0.01, step=1.0)
    r = st.sidebar.number_input("Risk-free rate (r)", value=0.05, step=0.01)
    sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01)
    T = st.sidebar.number_input("Time to maturity (T, in years)", value=1.0, step=0.1)
    barrier = st.sidebar.number_input("Barrier level (B)", value=110.0, min_value=0.0, step=1.0)
    M = st.sidebar.slider("Number of space steps (M)", min_value=50, max_value=100000, value=100)
    N = st.sidebar.slider("Number of time steps (N)", min_value=50, max_value=100000, value=100)

    st.sidebar.write("### Numerical Scheme")
    method = st.sidebar.selectbox("Finite-difference method",
                                  ["forward_euler", "backward_euler", "crank_nicolson"])

    # Construct the spatial grid
    # We'll pick a max S large enough to contain possible movements, e.g. 3-5x barrier or S0.
    Smax = max(3*barrier, 3*S0, 3*K)
    Smin = 0.0
    S_grid = np.linspace(Smin, Smax, M)

    # We'll compute for all 8 barrier types:
    # up-and-out, up-and-in, down-and-out, down-and-in for call, and same for put.
    # Then we plot each on a single graph for demonstration, or we can break them out.
    results = {}

    barrier_variants = [
        ("up-and-out",  "up",    "out"),
        ("up-and-in",   "up",    "in"),
        ("down-and-out","down",  "out"),
        ("down-and-in", "down",  "in")
    ]
    option_types = ["call", "put"]

    for opt_type in option_types:
        for b_label, b_type, in_out in barrier_variants:
            label = f"{opt_type} {b_label}"
            if in_out == "out":
                U = price_knock_out_vanilla(method, S_grid, r, sigma, K, T, M, N,
                                            barrier, b_type, opt_type)
            else:
                U = price_knock_in_vanilla(method, S_grid, r, sigma, K, T, M, N,
                                           barrier, b_type, opt_type)
            # Interpolate to find the price at S0
            price_at_S0 = np.interp(S0, S_grid, U)
            results[label] = (S_grid.copy(), U.copy(), price_at_S0)

    # Display results
    st.header("Results at S0")
    for label, (s_arr, U_arr, price_s0) in results.items():
        st.write(f"**{label}**: Price at S0 = {price_s0:.4f}")

    # Plot
    st.header("Barrier Option Values vs. Underlying Grid")
    fig, ax = plt.subplots()
    for label, (s_arr, U_arr, _) in results.items():
        ax.plot(s_arr, U_arr, label=label)
    ax.set_xlabel("Underlying Price S")
    ax.set_ylabel("Option Value")
    ax.set_title("Barrier Option Values (by type)")
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    main()

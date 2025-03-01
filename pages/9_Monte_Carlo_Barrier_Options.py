
# import streamlit as st
# import numpy as np
# import plotly.graph_objects as go
# from scipy.stats import norm

# # Set page configuration
# #st.set_page_config(page_title="Barrier Option Pricing using Monte Carlo Simulation", layout="wide")
# st.set_page_config(page_title="Barrier Option Pricing using Monte Carlo Simulation")

# # Title and description
# st.title("Barrier Option Pricing using Monte Carlo Simulation")

# st.write("""
# This page allows you to price various types of barrier options using Monte Carlo simulations.
# Select the option type and input the required parameters to calculate the option price.
# """)

# # -----------------------------------------------------------------------------
# # 1) Black-Scholes price for a European Call and Put
# # -----------------------------------------------------------------------------
# def black_scholes_call_price(S, K, T, r, sigma):
#     """
#     Computes the standard Black-Scholes price for a European Call Option.
#     Assumes zero dividend yield for simplicity; extend as needed.
#     """
#     d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
#     return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# def black_scholes_put_price(S, K, T, r, sigma):
#     """
#     Computes the standard Black-Scholes price for a European Put Option.
#     Assumes zero dividend yield for simplicity; extend as needed.
#     """
#     d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
#     return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

# # -----------------------------------------------------------------------------
# # 2) Barrier Option Monte Carlo (with optional Antithetic Variates)
# # -----------------------------------------------------------------------------
# def monte_carlo_barrier_option(
#     S0, K, H, T, r, q, sigma, M, N, option_type, rebate=0,
#     use_antithetic=False
# ):
#     """
#     Prices a barrier option using Monte Carlo simulation, with the option
#     to use Antithetic Variates for variance reduction.

#     Parameters:
#     - S0: Initial stock price
#     - K: Strike price
#     - H: Barrier level
#     - T: Time to maturity
#     - r: Risk-free interest rate
#     - q: Dividend yield
#     - sigma: Volatility
#     - M: Number of time steps
#     - N: Number of simulation paths (interpreted as # of pairs if use_antithetic=True)
#     - option_type: Type of barrier option
#     - rebate: Rebate paid if the option is knocked out or not activated
#     - use_antithetic: if True, generate pairs (+z, -z) and average their payoffs

#     Returns:
#     - barrier_price: Monte Carlo-estimated barrier option price
#     - paths_to_plot: A subset of simulated paths (just the +z ones)
#     - payoffs: The final payoffs for each “pair” (length N)
#     - barrier_breached: Boolean array indicating if barrier was hit (for the +z path)
#     """
#     dt = T / M
#     disc_factor = np.exp(-r * T)

#     is_call = 'call' in option_type.lower()
#     is_up   = 'up'   in option_type.lower()
#     is_in   = 'in'   in option_type.lower()
#     is_out  = not is_in

#     # If using antithetic, interpret N as # of pairs => total 2*N paths
#     # We'll produce one final payoff per pair, by averaging +z and -z.
#     paths_to_plot = np.zeros((N, M + 1))   # only storing the +z path for plotting
#     payoffs = np.zeros(N)
#     barrier_breached = np.zeros(N, dtype=bool)

#     for i in range(N):
#         # Generate one draw from standard normal for each time step
#         # or vector of draws if you prefer. For simplicity, we'll do
#         # step-by-step below.
#         S_plus  = S0
#         S_minus = S0

#         # Track if barrier is hit on +z path or -z path
#         breach_plus  = False
#         breach_minus = False
#         paths_to_plot[i, 0] = S_plus  # store initial price
#         for t in range(1, M + 1):
#             z = np.random.standard_normal()
#             # +z path
#             S_plus = S_plus * np.exp((r - q - 0.5 * sigma**2)*dt + sigma*np.sqrt(dt)*z)
#             paths_to_plot[i, t] = S_plus  # store for plotting
#             if not breach_plus:
#                 if is_up and S_plus >= H:
#                     breach_plus = True
#                 elif not is_up and S_plus <= H:
#                     breach_plus = True

#             # -z path
#             z_antithetic = -z
#             S_minus = S_minus * np.exp((r - q - 0.5 * sigma**2)*dt + sigma*np.sqrt(dt)*z_antithetic)
#             if not breach_minus:
#                 if is_up and S_minus >= H:
#                     breach_minus = True
#                 elif not is_up and S_minus <= H:
#                     breach_minus = True

#         # Now compute payoffs from +z and -z path
#         payoff_plus  = 0.0
#         payoff_minus = 0.0

#         # “Knock-in” => payoff if barrier was hit
#         # “Knock-out” => payoff if barrier was NOT hit
#         # Rebate if not “activated” or “knocked out,” etc.
#         if is_in:
#             # Option only exists if barrier was breached
#             if breach_plus:
#                 payoff_plus = max((S_plus - K), 0) if is_call else max((K - S_plus), 0)
#             else:
#                 payoff_plus = rebate

#             if breach_minus:
#                 payoff_minus = max((S_minus - K), 0) if is_call else max((K - S_minus), 0)
#             else:
#                 payoff_minus = rebate
#         else:
#             # is_out
#             if not breach_plus:
#                 payoff_plus = max((S_plus - K), 0) if is_call else max((K - S_plus), 0)
#             else:
#                 payoff_plus = rebate

#             if not breach_minus:
#                 payoff_minus = max((S_minus - K), 0) if is_call else max((K - S_minus), 0)
#             else:
#                 payoff_minus = rebate

#         # Antithetic average payoff
#         if use_antithetic:
#             avg_payoff = 0.5*(payoff_plus + payoff_minus)
#             payoffs[i] = avg_payoff
#         else:
#             # If not antithetic, we just use the +z path
#             payoffs[i] = payoff_plus

#         barrier_breached[i] = breach_plus  # track for the +z path

#     barrier_price = disc_factor * np.mean(payoffs)
#     return barrier_price, paths_to_plot, payoffs, barrier_breached

# # -----------------------------------------------------------------------------
# # 3) Streamlit UI
# # -----------------------------------------------------------------------------
# st.sidebar.header("Input Parameters")

# option_type = st.sidebar.selectbox(
#     "Select Barrier Option Type",
#     [
#         'Up-and-Out Call',
#         'Down-and-Out Call',
#         'Up-and-Out Put',
#         'Down-and-Out Put',
#         'Up-and-In Call',
#         'Down-and-In Call',
#         'Up-and-In Put',
#         'Down-and-In Put',
#     ]
# )

# S0 = st.sidebar.number_input("Initial Stock Price (S0)", min_value=0.01, value=100.0, step=1.0)
# K = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=100.0, step=1.0)
# H = st.sidebar.number_input("Barrier Level (H)", min_value=0.01, value=110.0, step=1.0)
# T = st.sidebar.number_input("Time to Maturity (T in years)", min_value=0.01, value=1.0, step=0.1)
# r = st.sidebar.number_input("Risk-Free Interest Rate (r)", min_value=0.0, value=0.05, step=0.01, format="%.4f")
# q = st.sidebar.number_input("Dividend Yield (q)", min_value=0.0, value=0.02, step=0.01, format="%.4f")
# sigma = st.sidebar.number_input("Volatility (σ in %)", min_value=0.01, value=20.0, step=0.1, format="%.2f") / 100
# M = st.sidebar.number_input("Number of Time Steps (M)", min_value=1, value=100, step=1)
# N = st.sidebar.number_input("Number of Simulation Paths (N)", min_value=100, value=10000, step=100)
# rebate = st.sidebar.number_input("Rebate", min_value=0.0, value=0.0, step=1.0)

# use_control_variate = st.sidebar.checkbox("Use Control Variate", value=False)
# use_antithetic      = st.sidebar.checkbox("Use Antithetic Variates", value=False)

# # Button to calculate
# if st.sidebar.button("Calculate Option Price"):

#     # Run the MC (with or without antithetic)
#     barrier_price, paths, payoffs_barrier, barrier_breached = monte_carlo_barrier_option(
#         S0, K, H, T, r, q, sigma, int(M), int(N), option_type,
#         rebate=rebate,
#         use_antithetic=use_antithetic
#     )
    
#     st.subheader(f"Monte Carlo Price: ${barrier_price:.4f} {'(Antithetic)' if use_antithetic else ''}")

#     # -------------------------
#     # Control Variate logic
#     # -------------------------
#     is_call = 'call' in option_type.lower()
#     if use_control_variate:
#         if is_call:
#             # Vanilla call
#             vanilla_analytic = black_scholes_call_price(S0, K, T, r, sigma)
#             # Simulated vanilla call payoff from final stock prices in `paths`
#             # (the +z path)
#             final_prices = paths[:, -1]
#             vanilla_payoffs = np.maximum(final_prices - K, 0.0)
#             vanilla_mc_est = np.exp(-r * T) * np.mean(vanilla_payoffs)
#             # CV formula
#             barrier_cv_price = barrier_price - vanilla_mc_est + vanilla_analytic

#             st.subheader("(Control Variate) with Vanilla Call")
#             st.write(f"Vanilla Call (Analytic) = **${vanilla_analytic:,.4f}**")
#             st.write(f"Vanilla Call (Simulated) = **${vanilla_mc_est:,.4f}**")
#             st.write(f"**Barrier Call (CV Estimate)** = **${barrier_cv_price:,.4f}**")

#         else:
#             # Vanilla put
#             vanilla_analytic_put = black_scholes_put_price(S0, K, T, r, sigma)
#             final_prices = paths[:, -1]
#             vanilla_put_payoffs = np.maximum(K - final_prices, 0.0)
#             vanilla_put_mc_est = np.exp(-r * T) * np.mean(vanilla_put_payoffs)
#             barrier_cv_price_put = barrier_price - vanilla_put_mc_est + vanilla_analytic_put

#             st.subheader("(Control Variate) with Vanilla Put")
#             st.write(f"Vanilla Put (Analytic) = **${vanilla_analytic_put:,.4f}**")
#             st.write(f"Vanilla Put (Simulated) = **${vanilla_put_mc_est:,.4f}**")
#             st.write(f"**Barrier Put (CV Estimate)** = **${barrier_cv_price_put:,.4f}**")

#     # -------------------------
#     # Plot some of the +z paths
#     # -------------------------
#     #'st.write("### Simulated Price Paths (Showing +z only)")
#     num_paths_to_plot = min(500, int(N))  # limit paths for performance
#     time_grid = np.linspace(0, T, M + 1)
#     fig = go.Figure()

#     for i in range(num_paths_to_plot):
#         path = paths[i]
#         color = 'green' if payoffs_barrier[i] > 0 else 'red'
#         fig.add_trace(go.Scatter(
#             x=time_grid, y=path,
#             mode='lines',
#             line=dict(color=color, width=1),
#             opacity=0.5
#         ))

#     # Add barrier line
#     fig.add_trace(go.Scatter(
#         x=[0, T], y=[H, H], mode='lines',
#         line=dict(color='blue', dash='dash'),
#         name='Barrier Level'
#     ))

#     fig.update_layout(
#             title={
#         "text": "Simulated Stock Price Paths",
#         "x": 0.5,         # centres the title horizontally
#         "xanchor": "center"
#     },
#         xaxis_title='Time (Years)',
#         yaxis_title='Stock Price',
#         showlegend=False
#     )
#     st.plotly_chart(fig, use_container_width=True)

# else:
#     st.info("Please input parameters and click 'Calculate Option Price' to compute the option price.")


import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# Set page configuration
st.set_page_config(page_title="Barrier Option Pricing using Monte Carlo Simulation", layout="wide")
st.title("Barrier Option Pricing using Monte Carlo Simulation")
st.write("""
This application prices various types of barrier options using Monte Carlo simulation.
You can choose to run the standard simulation or apply variance reduction techniques 
(antithetic variates and/or control variates) separately.
""")

# -----------------------------------------------------------------------------
# 1) Black-Scholes Price Functions for Vanilla Options
# -----------------------------------------------------------------------------
def black_scholes_call_price(S, K, T, r, sigma):
    """
    Computes the Black–Scholes price for a European Call Option.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put_price(S, K, T, r, sigma):
    """
    Computes the Black–Scholes price for a European Put Option.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# -----------------------------------------------------------------------------
# 2) Monte Carlo Barrier Option Pricing Function (with variance reduction options)
# -----------------------------------------------------------------------------
def monte_carlo_barrier_option_combined(S0, K, H, T, r, q, sigma, M, N, option_type, rebate=0,
                                        use_antithetic=False, use_control_variate=False):
    """
    Prices a barrier option using Monte Carlo simulation.
    
    The function allows for variance reduction via antithetic variates and/or control variates.
    
    Parameters:
      S0          : Initial stock price
      K           : Strike price
      H           : Barrier level
      T           : Time to maturity
      r           : Risk-free interest rate
      q           : Dividend yield
      sigma       : Volatility
      M           : Number of time steps
      N           : Number of simulation pairs (if antithetic is used, total paths = 2*N)
      option_type : String describing the barrier option type (e.g., 'Up-and-In Call')
      rebate      : Rebate paid if the option is knocked out or not activated
      use_antithetic      : If True, use antithetic variates (paired paths)
      use_control_variate : If True, adjust the barrier price using a control variate (vanilla option)
      
    Returns:
      barrier_price_cv : Adjusted barrier option price (if control variate is used)
      barrier_price_mc : Raw Monte Carlo barrier option price
      vanilla_mc       : Monte Carlo estimate for the corresponding vanilla option
    """
    dt = T / M
    disc_factor = np.exp(-r * T)
    
    is_call = 'call' in option_type.lower()
    is_up = 'up' in option_type.lower()
    is_in = 'in' in option_type.lower()
    
    payoffs = np.zeros(N)
    vanilla_payoffs = np.zeros(N)
    
    for i in range(N):
        S_plus = S0
        S_minus = S0
        breach_plus = False
        breach_minus = False
        
        for t in range(1, M + 1):
            z = np.random.standard_normal()
            # +z path update
            S_plus = S_plus * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            if not breach_plus:
                if (is_up and S_plus >= H) or ((not is_up) and S_plus <= H):
                    breach_plus = True
            # If antithetic variates are used, update the -z path
            if use_antithetic:
                z_antithetic = -z
                S_minus = S_minus * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z_antithetic)
                if not breach_minus:
                    if (is_up and S_minus >= H) or ((not is_up) and S_minus <= H):
                        breach_minus = True
            else:
                # Otherwise, simply use the +z path for both (i.e. no antithetic pairing)
                S_minus = S_plus
                breach_minus = breach_plus
        
        # Compute payoffs for barrier option based on option type
        if is_in:
            # Knock-in: option activates only if barrier is breached
            payoff_plus = max(S_plus - K, 0) if is_call and breach_plus else (max(K - S_plus, 0) if (not is_call) and breach_plus else rebate)
            payoff_minus = max(S_minus - K, 0) if is_call and breach_minus else (max(K - S_minus, 0) if (not is_call) and breach_minus else rebate)
        else:
            # Knock-out: option valid only if barrier is never breached
            payoff_plus = max(S_plus - K, 0) if is_call and (not breach_plus) else (max(K - S_plus, 0) if (not is_call) and (not breach_plus) else rebate)
            payoff_minus = max(S_minus - K, 0) if is_call and (not breach_minus) else (max(K - S_minus, 0) if (not is_call) and (not breach_minus) else rebate)
        
        if use_antithetic:
            avg_payoff = 0.5 * (payoff_plus + payoff_minus)
        else:
            avg_payoff = payoff_plus
        
        payoffs[i] = avg_payoff
        # For control variate: compute vanilla option payoff from the +z path
        vanilla_payoffs[i] = max(S_plus - K, 0) if is_call else max(K - S_plus, 0)
    
    barrier_price_mc = disc_factor * np.mean(payoffs)
    vanilla_mc = disc_factor * np.mean(vanilla_payoffs)
    
    if use_control_variate:
        if is_call:
            vanilla_analytic = black_scholes_call_price(S0, K, T, r, sigma)
        else:
            vanilla_analytic = black_scholes_put_price(S0, K, T, r, sigma)
        barrier_price_cv = barrier_price_mc - vanilla_mc + vanilla_analytic
    else:
        barrier_price_cv = barrier_price_mc
        
    return barrier_price_cv, barrier_price_mc, vanilla_mc

# -----------------------------------------------------------------------------
# 3) Streamlit User Interface
# -----------------------------------------------------------------------------
st.sidebar.header("Input Parameters")

# Select the Monte Carlo method
method = st.sidebar.radio("Select Monte Carlo Method", 
    options=["Standard MC", "Antithetic MC", "Control Variate MC", "Combined MC"])

# Map method selection to variance reduction flags
if method == "Standard MC":
    use_antithetic = False
    use_control_variate = False
elif method == "Antithetic MC":
    use_antithetic = True
    use_control_variate = False
elif method == "Control Variate MC":
    use_antithetic = False
    use_control_variate = True
elif method == "Combined MC":
    use_antithetic = True
    use_control_variate = True

option_type = st.sidebar.selectbox(
    "Select Barrier Option Type",
    ['Up-and-Out Call', 'Down-and-Out Call', 'Up-and-Out Put', 'Down-and-Out Put',
     'Up-and-In Call', 'Down-and-In Call', 'Up-and-In Put', 'Down-and-In Put']
)
S0 = st.sidebar.number_input("Initial Stock Price (S0)", min_value=0.01, value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=100.0, step=1.0)
H = st.sidebar.number_input("Barrier Level (H)", min_value=0.01, value=110.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (T in years)", min_value=0.01, value=1.0, step=0.1)
r = st.sidebar.number_input("Risk-Free Interest Rate (r)", min_value=0.0, value=0.05, step=0.01, format="%.4f")
q = st.sidebar.number_input("Dividend Yield (q)", min_value=0.0, value=0.02, step=0.01, format="%.4f")
sigma = st.sidebar.number_input("Volatility (σ in %)", min_value=0.01, value=20.0, step=0.1, format="%.2f") / 100
M = st.sidebar.number_input("Number of Time Steps (M)", min_value=1, value=100, step=1)
N = st.sidebar.number_input("Number of Simulation Pairs (N)", min_value=100, value=10000, step=100)
rebate = st.sidebar.number_input("Rebate", min_value=0.0, value=0.0, step=1.0)

if st.sidebar.button("Calculate Option Price"):
    barrier_cv, barrier_mc, vanilla_mc = monte_carlo_barrier_option_combined(
        S0, K, H, T, r, q, sigma, int(M), int(N), option_type, rebate,
        use_antithetic=use_antithetic, use_control_variate=use_control_variate
    )
    st.subheader(f"Method Selected: {method}")
    st.subheader(f"Raw Monte Carlo Barrier Option Price: ${barrier_mc:.4f}")
    if use_control_variate:
        st.subheader(f"Vanilla Option Price (MC): ${vanilla_mc:.4f}")
        st.subheader(f"Adjusted Barrier Option Price (Combined): ${barrier_cv:.4f}")
    else:
        st.subheader(f"Barrier Option Price: ${barrier_mc:.4f}")
else:
    st.info("Please input parameters and click 'Calculate Option Price' to compute the option price.")

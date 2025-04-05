
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import norm
import plotly.graph_objects as go
# ------------------------------
# 1) Helper functions
# ------------------------------

def calc_d1(S0, K, r, q, sigma, T):
    return (np.log(S0 / K) + (r - q + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))

def calc_d2(S0, K, r, q, sigma, T):
    return calc_d1(S0, K, r, q, sigma, T) - sigma*np.sqrt(T)

def calc_c(S0, K, r, q, sigma, T):
    d1 = calc_d1(S0, K, r, q, sigma, T)
    d2 = calc_d2(S0, K, r, q, sigma, T)
    return (S0 * np.exp(-q*T)*norm.cdf(d1)
            - K * np.exp(-r*T)*norm.cdf(d2))

def calc_p(S0, K, r, q, sigma, T):
    d1 = calc_d1(S0, K, r, q, sigma, T)
    d2 = calc_d2(S0, K, r, q, sigma, T)
    return (K*np.exp(-r*T)*norm.cdf(-d2)
            - S0*np.exp(-q*T)*norm.cdf(-d1))

def calc_lambda(r, q, sigma):
    return (r - q + 0.5*sigma**2) / (sigma**2)

def calc_y(H, S0, K, T, sigma, r, q):
    lam = calc_lambda(r, q, sigma)
    return (np.log((H**2)/(S0*K)) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

def calc_x1(S0, H, T, sigma, r, q):
    lam = calc_lambda(r, q, sigma)
    return (np.log(S0/H) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

def calc_y1(S0, H, T, sigma, r, q):
    lam = calc_lambda(r, q, sigma)
    return (np.log(H/S0) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

# ------------------------------
# 2) Barrier Option Closed-Form
# ------------------------------

def barrier_option_price(S0, K, T, r, q, sigma, H, option_type):
    """
    Returns the price of a barrier option (various knock-in/out types).
    Matches standard formulas from texts like Hull.
    """
    x1 = calc_x1(S0, H, T, sigma, r, q)
    y1 = calc_y1(S0, H, T, sigma, r, q)
    c = calc_c(S0, K, r, q, sigma, T)
    p = calc_p(S0, K, r, q, sigma, T)
    lam = calc_lambda(r, q, sigma)
    y  = calc_y(H, S0, K, T, sigma, r, q)

    # --------------------------------
    # Down-and-in Call
    # --------------------------------
    if option_type == 'down-and-in call' and H <= K:
        cdi = (S0 * np.exp(-q*T) * (H/S0)**(2*lam) * norm.cdf(y)
               - K * np.exp(-r*T) * (H/S0)**(2*lam - 2)
                 * norm.cdf(y - sigma*np.sqrt(T)))
        return cdi

    elif option_type == 'down-and-in call' and H >= K:
        term1 = S0*np.exp(-q*T)*norm.cdf(x1)
        term2 = K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
        term3 = S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(y1)
        term4 = K*np.exp(-r*T)*(H/S0)**(2*lam - 2)*norm.cdf(y1 - sigma*np.sqrt(T))
        cdo   = term1 - term2 - term3 + term4
        cdo   = max(cdo, 0.0)
        cdi   = c - cdo
        return cdi

    # --------------------------------
    # Down-and-out Call
    # --------------------------------
    elif option_type == 'down-and-out call' and H <= K:
        cdi = (S0 * np.exp(-q*T) * (H/S0)**(2*lam) * norm.cdf(y)
               - K * np.exp(-r*T) * (H/S0)**(2*lam - 2)
                 * norm.cdf(y - sigma*np.sqrt(T)))
        cdo = c - cdi
        return cdo

    elif option_type == 'down-and-out call' and H >= K:
        term1 = S0 * np.exp(-q*T)*norm.cdf(x1)
        term2 = K  * np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
        term3 = S0 * np.exp(-q*T)*((H/S0)**(2*lam))*norm.cdf(y1)
        term4 = K  * np.exp(-r*T)*((H/S0)**(2*lam - 2))*norm.cdf(y1 - sigma*np.sqrt(T))
        cdo   = term1 - term2 - term3 + term4
        return max(cdo, 0.0)

    # --------------------------------
    # Up-and-in Call
    # --------------------------------
    elif option_type == 'up-and-in call' and H > K:
        cui = (S0*np.exp(-q*T)*norm.cdf(x1)
               - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
               - S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(-y) - norm.cdf(-y1))
               + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
                 * (norm.cdf(-y + sigma*np.sqrt(T))
                    - norm.cdf(-y1 + sigma*np.sqrt(T))))
        return cui

    elif option_type == 'up-and-in call' and H <= K:
        return c

    # --------------------------------
    # Up-and-out Call
    # --------------------------------
    elif option_type == 'up-and-out call' and H <= K:
        return 0.0

    elif option_type == 'up-and-out call' and H > K:
        cui = (S0*np.exp(-q*T)*norm.cdf(x1)
               - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
               - S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(-y) - norm.cdf(-y1))
               + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
                 * (norm.cdf(-y + sigma*np.sqrt(T))
                    - norm.cdf(-y1 + sigma*np.sqrt(T))))
        return c - cui

    # --------------------------------
    # Up-and-in Put
    # --------------------------------
    elif option_type == 'up-and-in put' and H >= K:
        pui = (-S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y)
               + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
                 * norm.cdf(-y + sigma*np.sqrt(T)))
        return pui

    elif option_type == 'up-and-in put' and H <= K:
        return p

    # --------------------------------
    # Up-and-out Put
    # --------------------------------
    elif option_type == 'up-and-out put' and H >= K:
        pui = (-S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y)
               + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
                 * norm.cdf(-y + sigma*np.sqrt(T)))
        puo = p - pui
        return puo

    elif option_type == 'up-and-out put' and H <= K:
        puo = (
            -S0*np.exp(-q*T)*norm.cdf(-x1)
            + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
            + S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y1)
            - K*np.exp(-r*T)*(H/S0)**(2*lam - 2)*norm.cdf(-y1 + sigma*np.sqrt(T))
        )
        return max(puo, 0.0)

    # --------------------------------
    # Down-and-in Put
    # --------------------------------
    elif option_type == 'down-and-in put' and H > K:
        return p

    elif option_type == 'down-and-in put' and H < K:
        pdi = (
            -S0*np.exp(-q*T)*norm.cdf(-x1)
            + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
            + S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(y) - norm.cdf(y1))
            - K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
              * (norm.cdf(y - sigma*np.sqrt(T))
                 - norm.cdf(y1 - sigma*np.sqrt(T)))
        )
        return pdi

    # --------------------------------
    # Down-and-out Put
    # --------------------------------
    elif option_type == 'down-and-out put' and H > K:
        return 0.0

    elif option_type == 'down-and-out put' and H < K:
        pdi = (
            -S0*np.exp(-q*T)*norm.cdf(-x1)
            + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
            + S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(y) - norm.cdf(y1))
            - K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
              * (norm.cdf(y - sigma*np.sqrt(T))
                 - norm.cdf(y1 - sigma*np.sqrt(T)))
        )
        pdo = p - pdi
        return pdo

    return None

# --------------------------------------------------------------------------
# Black–Scholes for vanilla (for control variate)
# --------------------------------------------------------------------------
def black_scholes_call(S0, K, T, r, sigma):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def black_scholes_put(S0, K, T, r, sigma):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)

# --------------------------------------------------------------------------
# 3) Monte Carlo Approaches
# --------------------------------------------------------------------------

def mc_barrier_option_plain(S0, K, H, T, r, q, sigma, N, M, option_type):
    dt   = T / M
    disc = np.exp(-r*T)
    is_call = 'call' in option_type.lower()
    is_in   = 'in' in option_type.lower()
    is_up   = 'up' in option_type.lower()

    payoffs = np.zeros(N)
    for i in range(N):
        S_path = S0
        barrier_breached = False
        for _ in range(M):
            z = np.random.randn()
            S_path *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
            if is_up and S_path >= H:
                barrier_breached = True
            elif (not is_up) and S_path <= H:
                barrier_breached = True

        if is_in:
            if barrier_breached:
                payoff = max(S_path-K, 0) if is_call else max(K-S_path, 0)
            else:
                payoff = 0.0
        else:
            if barrier_breached:
                payoff = 0.0
            else:
                payoff = max(S_path-K, 0) if is_call else max(K-S_path, 0)
        payoffs[i] = payoff

    return disc*np.mean(payoffs)

def mc_barrier_option_cv(S0, K, H, T, r, q, sigma, N, M, option_type):
    dt   = T / M
    disc = np.exp(-r*T)
    is_call = 'call' in option_type.lower()
    is_in   = 'in' in option_type.lower()
    is_up   = 'up' in option_type.lower()

    # pick correct vanilla
    if is_call:
        vanilla_analytic = black_scholes_call(S0, K, T, r, sigma)
    else:
        vanilla_analytic = black_scholes_put(S0, K, T, r, sigma)

    barrier_payoffs = np.zeros(N)
    vanilla_payoffs = np.zeros(N)
    for i in range(N):
        S_path = S0
        barrier_breached = False
        for _ in range(M):
            z = np.random.randn()
            S_path *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
            if is_up and S_path >= H:
                barrier_breached = True
            elif (not is_up) and S_path <= H:
                barrier_breached = True

        # Barrier payoff
        if is_in:
            payoff_barrier = (
                max(S_path-K, 0) if (is_call and barrier_breached) else
                max(K-S_path, 0) if (not is_call and barrier_breached) else 0.0
            )
        else:
            payoff_barrier = (
                max(S_path-K, 0) if (is_call and not barrier_breached) else
                max(K-S_path, 0) if (not is_call and not barrier_breached) else 0.0
            )
        barrier_payoffs[i] = payoff_barrier

        # Vanilla payoff
        payoff_vanilla = max(S_path-K, 0) if is_call else max(K-S_path, 0)
        vanilla_payoffs[i] = payoff_vanilla

    mc_barrier = disc*np.mean(barrier_payoffs)
    mc_vanilla = disc*np.mean(vanilla_payoffs)
    mc_cv = mc_barrier + (vanilla_analytic - mc_vanilla)
    return mc_cv

def mc_barrier_option_av(S0, K, H, T, r, q, sigma, N, M, option_type):
    dt   = T / M
    disc = np.exp(-r*T)
    is_call = 'call' in option_type.lower()
    is_in   = 'in' in option_type.lower()
    is_up   = 'up' in option_type.lower()

    payoffs = np.zeros(N)
    for i in range(N):
        S_plus  = S0
        S_minus = S0
        breach_plus  = False
        breach_minus = False

        for _ in range(M):
            z = np.random.randn()
            # +z path
            S_plus *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
            if is_up and S_plus >= H:
                breach_plus = True
            elif (not is_up) and S_plus <= H:
                breach_plus = True

            # -z path
            S_minus *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*(-z))
            if is_up and S_minus >= H:
                breach_minus = True
            elif (not is_up) and S_minus <= H:
                breach_minus = True

        if is_in:
            payoff_plus = (
                max(S_plus-K, 0) if (is_call and breach_plus) else
                max(K-S_plus, 0) if (not is_call and breach_plus) else 0.0
            )
            payoff_minus = (
                max(S_minus-K, 0) if (is_call and breach_minus) else
                max(K-S_minus, 0) if (not is_call and breach_minus) else 0.0
            )
        else:
            payoff_plus = (
                max(S_plus-K, 0) if (is_call and not breach_plus) else
                max(K-S_plus, 0) if (not is_call and not breach_plus) else 0.0
            )
            payoff_minus = (
                max(S_minus-K, 0) if (is_call and not breach_minus) else
                max(K-S_minus, 0) if (not is_call and not breach_minus) else 0.0
            )

        payoffs[i] = 0.5*(payoff_plus + payoff_minus)

    return disc*np.mean(payoffs)

def mc_barrier_option_combined(S0, K, H, T, r, q, sigma, N, M, option_type):
    dt   = T / M
    disc = np.exp(-r*T)
    is_call = 'call' in option_type.lower()
    is_up   = 'up' in option_type.lower()
    is_in   = 'in' in option_type.lower()

    # For the control variate
    if is_call:
        vanilla_analytic = black_scholes_call(S0, K, T, r, sigma)
    else:
        vanilla_analytic = black_scholes_put(S0, K, T, r, sigma)

    barrier_payoffs = np.zeros(N)
    vanilla_payoffs = np.zeros(N)
    for i in range(N):
        S_plus  = S0
        S_minus = S0
        breach_plus  = False
        breach_minus = False
        for _ in range(M):
            z = np.random.randn()
            S_plus *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
            if is_up and S_plus >= H:
                breach_plus = True
            elif (not is_up) and S_plus <= H:
                breach_plus = True

            S_minus *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*(-z))
            if is_up and S_minus >= H:
                breach_minus = True
            elif (not is_up) and S_minus <= H:
                breach_minus = True

        # barrier payoff
        if is_in:
            payoff_plus = (
                max(S_plus-K, 0) if (is_call and breach_plus) else
                max(K-S_plus, 0) if (not is_call and breach_plus) else 0.0
            )
            payoff_minus= (
                max(S_minus-K, 0) if (is_call and breach_minus) else
                max(K-S_minus, 0) if (not is_call and breach_minus) else 0.0
            )
        else:
            payoff_plus = (
                max(S_plus-K, 0) if (is_call and not breach_plus) else
                max(K-S_plus, 0) if (not is_call and not breach_plus) else 0.0
            )
            payoff_minus= (
                max(S_minus-K, 0) if (is_call and not breach_minus) else
                max(K-S_minus, 0) if (not is_call and not breach_minus) else 0.0
            )

        avg_barrier = 0.5*(payoff_plus + payoff_minus)
        barrier_payoffs[i] = avg_barrier

        # For control variate, we only need the vanilla payoff from S_plus
        if is_call:
            payoff_vanilla = max(S_plus-K,0)
        else:
            payoff_vanilla = max(K-S_plus,0)
        vanilla_payoffs[i] = payoff_vanilla

    barrier_raw = disc*np.mean(barrier_payoffs)
    vanilla_mc  = disc*np.mean(vanilla_payoffs)
    barrier_combined = barrier_raw + (vanilla_analytic - vanilla_mc)
    return barrier_combined


# st.set_page_config(page_title="Barrier Option Pricing: Single Plot Comparison")

# st.sidebar.header("Input Parameters")
# option_type = st.sidebar.selectbox(
#     "Barrier Option Type",
#     [
#         "down-and-in call",
#         "down-and-out call",
#         "down-and-in put",
#         "down-and-out put",
#         "up-and-in call",
#         "up-and-out call",
#         "up-and-in put",
#         "up-and-out put",
#     ]
# )
# S0 = st.sidebar.number_input("Spot (S0)", value=50.0)
# K  = st.sidebar.number_input("Strike (K)", value=50.0)
# H  = st.sidebar.number_input("Barrier (H)", value=40.0)
# T  = st.sidebar.number_input("Maturity (T)", value=1.0)
# r  = st.sidebar.number_input("Risk-Free Rate (r)", value=0.1)
# q  = st.sidebar.number_input("Dividend Yield (q)", value=0.00)
# sigma = st.sidebar.number_input("Volatility (σ)", value=0.2)

# st.sidebar.write("MC Steps (M) and # of Paths (N)")

# # Multiple M values
# default_M = "50,100,200"
# M_str = st.sidebar.text_input("Comma-separated M values:", default_M)
# M_list = []
# try:
#     M_list = sorted([int(x.strip()) for x in M_str.split(",") if int(x.strip()) > 0])
# except:
#     st.error("Invalid input for M values. Please enter positive integers separated by commas.")

# # Multiple N values
# default_N = "100,1000,10000"
# N_str = st.sidebar.text_input("Comma-separated N values:", default_N)
# N_list = []
# try:
#     N_list = sorted([int(x.strip()) for x in N_str.split(",") if int(x.strip()) > 0])
# except:
#     st.error("Invalid input for N values. Please enter positive integers separated by commas.")

# tab1, tab2 = st.tabs(["Closed-Form", "MC Analysis"])

# with tab1:
#     st.title("Closed-Form (Analytical) Price")
#     cf_price = barrier_option_price(S0, K, T, r, q, sigma, H, option_type)
#     if cf_price is not None:
#         st.subheader(f"{option_type.capitalize()} Price = {cf_price:.4f}")
#     else:
#         st.write("No closed-form formula implemented or invalid parameters.")

# # Initialize simulation results in session_state if not already present.
# if "df_results" not in st.session_state:
#     st.session_state.df_results = None
# if "analytical_price" not in st.session_state:
#     st.session_state.analytical_price = None

# with tab2:
#     st.title("Monte Carlo Comparison")
    
#     # When the user clicks "Run MC", store results in session state
#     if st.button("Run MC"):
#         analytical_price = barrier_option_price(S0, K, T, r, q, sigma, H, option_type)
#         st.session_state.analytical_price = analytical_price

#         rows = []
#         for m_val in M_list:
#             for n_val in N_list:
#                 mc_plain = mc_barrier_option_plain(S0, K, H, T, r, q, sigma, n_val, m_val, option_type)
#                 mc_cv    = mc_barrier_option_cv(S0, K, H, T, r, q, sigma, n_val, m_val, option_type)
#                 mc_av    = mc_barrier_option_av(S0, K, H, T, r, q, sigma, n_val, m_val, option_type)
#                 mc_comb  = mc_barrier_option_combined(S0, K, H, T, r, q, sigma, n_val, m_val, option_type)
#                 rows.append({
#                     "M": m_val,
#                     "N": n_val,
#                     "Plain": mc_plain,
#                     "CV": mc_cv,
#                     "AV": mc_av,
#                     "Combined": mc_comb,
#                     "Plain Error": np.abs(mc_plain - analytical_price) if analytical_price is not None else np.nan,
#                     "CV Error": np.abs(mc_cv - analytical_price) if analytical_price is not None else np.nan,
#                     "AV Error": np.abs(mc_av - analytical_price) if analytical_price is not None else np.nan,
#                     "Combined Error": np.abs(mc_comb - analytical_price) if analytical_price is not None else np.nan
#                 })

#         df = pd.DataFrame(rows)
#         st.session_state.df_results = df

#     # If simulation has been run, display the results.
#     if st.session_state.df_results is not None:
#         df = st.session_state.df_results
#         analytical_price = st.session_state.analytical_price
        
#         # Pivot tables for each method (Price Estimates)
#         st.write("## Pivot Tables by Method (Price Estimates)")
#         st.write("### Plain MC")
#         df_plain = df.pivot(index="N", columns="M", values="Plain")
#         st.dataframe(df_plain.style.format("{:.4f}"), use_container_width=True)

#         st.write("### Control Variate (CV)")
#         df_cv = df.pivot(index="N", columns="M", values="CV")
#         st.dataframe(df_cv.style.format("{:.4f}"), use_container_width=True)

#         st.write("### Antithetic Variate (AV)")
#         df_av = df.pivot(index="N", columns="M", values="AV")
#         st.dataframe(df_av.style.format("{:.4f}"), use_container_width=True)

#         st.write("### Combined (CV + AV)")
#         df_comb = df.pivot(index="N", columns="M", values="Combined")
#         st.dataframe(df_comb.style.format("{:.4f}"), use_container_width=True)

#         # Graph comparing option values for a selected M
#         st.write("## Option Value Comparison (Variance Reduction Techniques)")
#         df_values = df[["M", "N", "Plain", "CV", "AV", "Combined"]].copy()
#         df_melt_values = df_values.melt(id_vars=["M", "N"],
#                                         value_vars=["Plain", "CV", "AV", "Combined"],
#                                         var_name="Method", value_name="OptionValue")
#         selected_M = st.selectbox("Select M to visualize option value across N:", M_list, key="value_M")
#         df_plot_values = df_melt_values[df_melt_values["M"] == selected_M]
#         fig_value = px.line(
#             df_plot_values,
#             x="N",
#             y="OptionValue",
#             color="Method",
#             markers=True,
#             title=f"Option Value vs. N at M={selected_M}"
#         )
#         fig_value.update_layout(
#             title={"x":0.5, "xanchor":"center"},
#             legend_title_text="MC Method"
#         )
#         if analytical_price is not None:
#             fig_value.add_hline(
#                 y=analytical_price,
#                 line_dash="dash",
#                 line_color="red",
#                 annotation_text=f"Analytical = {analytical_price:.4f}",
#                 annotation_position="top right"
#             )
#         st.plotly_chart(fig_value, use_container_width=True)

#         # 3D Surface Plot for Absolute Error
#         st.write("## 3D Surface Plot: Absolute Error vs. N and M")
#         selected_error_method = st.selectbox(
#             "Select MC Method Error for 3D Visualization:", 
#             ["Plain Error", "CV Error", "AV Error", "Combined Error"],
#             key="error_method"
#         )
#         df_error_pivot = df.pivot(index="N", columns="M", values=selected_error_method)
#         x_vals = list(df_error_pivot.columns)  # Time Steps (M)
#         y_vals = list(df_error_pivot.index)     # Number of Paths (N)
#         z_vals = df_error_pivot.values          # 2D array of error values
#         fig_error_surface = go.Figure(data=[go.Surface(x=x_vals, y=y_vals, z=z_vals)])
#         fig_error_surface.update_layout(
#             title=f"3D Surface Plot of Absolute Error: {selected_error_method}",
#             scene=dict(
#                 xaxis_title="Time Steps (M)",
#                 yaxis_title="Number of Paths (N)",
#                 zaxis_title="Absolute Error"
#             )
#         )
#         st.plotly_chart(fig_error_surface, use_container_width=True)

#         # Full results table with highlighted minimum errors
#         st.write("## Full Results (Including Errors)")
#         def highlight_min_error(row):
#             error_cols = ["Plain Error", "CV Error", "AV Error", "Combined Error"]
#             is_min = row[error_cols] == row[error_cols].min()
#             return ["background-color: #ffff66" if v else "" for v in is_min]

#         df_style = df.style.format("{:.4f}") \
#                            .apply(highlight_min_error, axis=1, 
#                                   subset=["Plain Error", "CV Error", "AV Error", "Combined Error"])
#         st.dataframe(df_style, use_container_width=True)

#         if analytical_price is not None:
#             st.markdown(f"**Closed-Form Analytical Price:** `{analytical_price:.4f}`")
#     else:
#         st.info("Click 'Run MC' to compute and compare methods over different time steps (M) and number of paths (N).")

st.set_page_config(page_title="Barrier Option Pricing: Single Plot Comparison")

st.sidebar.header("Input Parameters")
option_type = st.sidebar.selectbox(
    "Barrier Option Type",
    [
        "down-and-in call",
        "down-and-out call",
        "down-and-in put",
        "down-and-out put",
        "up-and-in call",
        "up-and-out call",
        "up-and-in put",
        "up-and-out put",
    ]
)
S0 = st.sidebar.number_input("Spot (S0)", value=50.0)
K  = st.sidebar.number_input("Strike (K)", value=50.0)
H  = st.sidebar.number_input("Barrier (H)", value=40.0)
T  = st.sidebar.number_input("Maturity (T)", value=1.0)
r  = st.sidebar.number_input("Risk-Free Rate (r)", value=0.1)
q  = st.sidebar.number_input("Dividend Yield (q)", value=0.00)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2)

st.sidebar.write("MC Steps (M) and # of Paths (N)")

# Multiple M values
default_M = "50,100,200"
M_str = st.sidebar.text_input("Comma-separated M values:", default_M)
M_list = []
try:
    M_list = sorted([int(x.strip()) for x in M_str.split(",") if int(x.strip()) > 0])
except:
    st.error("Invalid input for M values. Please enter positive integers separated by commas.")

# Multiple N values
default_N = "100,1000,10000"
N_str = st.sidebar.text_input("Comma-separated N values:", default_N)
N_list = []
try:
    N_list = sorted([int(x.strip()) for x in N_str.split(",") if int(x.strip()) > 0])
except:
    st.error("Invalid input for N values. Please enter positive integers separated by commas.")

tab1, tab2 = st.tabs(["Closed-Form", "MC Analysis"])

with tab1:
    st.title("Closed-Form (Analytical) Price")
    cf_price = barrier_option_price(S0, K, T, r, q, sigma, H, option_type)
    if cf_price is not None:
        st.subheader(f"{option_type.capitalize()} Price = {cf_price:.4f}")
    else:
        st.write("No closed-form formula implemented or invalid parameters.")

# Initialize simulation results in session_state if not already present.
if "df_results" not in st.session_state:
    st.session_state.df_results = None
if "analytical_price" not in st.session_state:
    st.session_state.analytical_price = None

with tab2:
    st.title("Monte Carlo Comparison")
    
    # When the user clicks "Run MC", store results in session state
    if st.button("Run MC"):
        analytical_price = barrier_option_price(S0, K, T, r, q, sigma, H, option_type)
        st.session_state.analytical_price = analytical_price

        rows = []
        for m_val in M_list:
            for n_val in N_list:
                mc_plain = mc_barrier_option_plain(S0, K, H, T, r, q, sigma, n_val, m_val, option_type)
                mc_cv    = mc_barrier_option_cv(S0, K, H, T, r, q, sigma, n_val, m_val, option_type)
                mc_av    = mc_barrier_option_av(S0, K, H, T, r, q, sigma, n_val, m_val, option_type)
                mc_comb  = mc_barrier_option_combined(S0, K, H, T, r, q, sigma, n_val, m_val, option_type)
                rows.append({
                    "M": m_val,
                    "N": n_val,
                    "Plain": mc_plain,
                    "CV": mc_cv,
                    "AV": mc_av,
                    "Combined": mc_comb,
                    "Plain Error": np.abs(mc_plain - analytical_price) if analytical_price is not None else np.nan,
                    "CV Error": np.abs(mc_cv - analytical_price) if analytical_price is not None else np.nan,
                    "AV Error": np.abs(mc_av - analytical_price) if analytical_price is not None else np.nan,
                    "Combined Error": np.abs(mc_comb - analytical_price) if analytical_price is not None else np.nan
                })

        df = pd.DataFrame(rows)
        st.session_state.df_results = df

    # If simulation has been run, display the results.
    if st.session_state.df_results is not None:
        df = st.session_state.df_results
        analytical_price = st.session_state.analytical_price
        
        # Pivot tables for each method (Price Estimates)
        st.write("## Pivot Tables by Method (Price Estimates)")
        st.write("### Plain MC")
        df_plain = df.pivot(index="N", columns="M", values="Plain")
        st.dataframe(df_plain.style.format("{:.4f}"), use_container_width=True)

        st.write("### Control Variate (CV)")
        df_cv = df.pivot(index="N", columns="M", values="CV")
        st.dataframe(df_cv.style.format("{:.4f}"), use_container_width=True)

        st.write("### Antithetic Variate (AV)")
        df_av = df.pivot(index="N", columns="M", values="AV")
        st.dataframe(df_av.style.format("{:.4f}"), use_container_width=True)

        st.write("### Combined (CV + AV)")
        df_comb = df.pivot(index="N", columns="M", values="Combined")
        st.dataframe(df_comb.style.format("{:.4f}"), use_container_width=True)

        # Graph comparing option values for a selected M
        st.write("## Option Value Comparison (Variance Reduction Techniques)")
        df_values = df[["M", "N", "Plain", "CV", "AV", "Combined"]].copy()
        df_melt_values = df_values.melt(id_vars=["M", "N"],
                                        value_vars=["Plain", "CV", "AV", "Combined"],
                                        var_name="Method", value_name="OptionValue")
        selected_M = st.selectbox("Select M to visualize option value across N:", M_list, key="value_M")
        df_plot_values = df_melt_values[df_melt_values["M"] == selected_M]
        fig_value = px.line(
            df_plot_values,
            x="N",
            y="OptionValue",
            color="Method",
            markers=True,
            title=f"Option Value vs. N at M={selected_M}"
        )
        # Enhance 2D plot axes readability
        fig_value.update_layout(
            title={"x":0.5, "xanchor":"center", "font": {"size":24}},
            legend_title_text="MC Method",
            xaxis=dict(
                title="Number of Paths (N)",
                titlefont=dict(size=20, color='black'),
                tickfont=dict(size=16, color='black')
            ),
            yaxis=dict(
                title="Option Value",
                titlefont=dict(size=20, color='black'),
                tickfont=dict(size=16, color='black')
            )
        )
        if analytical_price is not None:
            fig_value.add_hline(
                y=analytical_price,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Analytical = {analytical_price:.4f}",
                annotation_position="top right"
            )
        st.plotly_chart(fig_value, use_container_width=True)

        # 3D Surface Plot for Absolute Error with enhanced axis readability
        st.write("## 3D Surface Plot: Absolute Error vs. N and M")
        selected_error_method = st.selectbox(
            "Select MC Method Error for 3D Visualization:", 
            ["Plain Error", "CV Error", "AV Error", "Combined Error"],
            key="error_method"
        )
        df_error_pivot = df.pivot(index="N", columns="M", values=selected_error_method)
        x_vals = list(df_error_pivot.columns)  # Time Steps (M)
        y_vals = list(df_error_pivot.index)     # Number of Paths (N)
        z_vals = df_error_pivot.values          # 2D array of error values
        fig_error_surface = go.Figure(data=[go.Surface(x=x_vals, y=y_vals, z=z_vals)])
        fig_error_surface.update_layout(
            title=f"3D Surface Plot of Absolute Error: {selected_error_method}",
            scene=dict(
                xaxis=dict(
                    title="Time Steps (M)",
                    titlefont=dict(size=20, color='black'),
                    tickfont=dict(size=16, color='black')
                ),
                yaxis=dict(
                    title="Number of Paths (N)",
                    titlefont=dict(size=20, color='black'),
                    tickfont=dict(size=16, color='black')
                ),
                zaxis=dict(
                    title="Absolute Error",
                    titlefont=dict(size=20, color='black'),
                    tickfont=dict(size=16, color='black')
                )
            )
        )
        st.plotly_chart(fig_error_surface, use_container_width=True)

        # Full results table with highlighted minimum errors
        st.write("## Full Results (Including Errors)")
        def highlight_min_error(row):
            error_cols = ["Plain Error", "CV Error", "AV Error", "Combined Error"]
            is_min = row[error_cols] == row[error_cols].min()
            return ["background-color: #ffff66" if v else "" for v in is_min]

        df_style = df.style.format("{:.4f}") \
                           .apply(highlight_min_error, axis=1, 
                                  subset=["Plain Error", "CV Error", "AV Error", "Combined Error"])
        st.dataframe(df_style, use_container_width=True)

        if analytical_price is not None:
            st.markdown(f"**Closed-Form Analytical Price:** `{analytical_price:.4f}`")
    else:
        st.info("Click 'Run MC' to compute and compare methods over different time steps (M) and number of paths (N).")
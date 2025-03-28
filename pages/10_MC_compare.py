# # # import streamlit as st
# # # import numpy as np
# # # import pandas as pd
# # # import plotly.express as px
# # # from scipy.stats import norm

# # # # ------------------------------
# # # # 1) Helper functions
# # # # ------------------------------

# # # def calc_d1(S0, K, r, q, sigma, T):
# # #     return (np.log(S0 / K) + (r - q + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))

# # # def calc_d2(S0, K, r, q, sigma, T):
# # #     return calc_d1(S0, K, r, q, sigma, T) - sigma * np.sqrt(T)

# # # def calc_c(S0, K, r, q, sigma, T):
# # #     d1 = calc_d1(S0, K, r, q, sigma, T)
# # #     d2 = calc_d2(S0, K, r, q, sigma, T)
# # #     return (S0 * np.exp(-q*T)*norm.cdf(d1)
# # #             - K * np.exp(-r*T)*norm.cdf(d2))

# # # def calc_p(S0, K, r, q, sigma, T):
# # #     d1 = calc_d1(S0, K, r, q, sigma, T)
# # #     d2 = calc_d2(S0, K, r, q, sigma, T)
# # #     return (K * np.exp(-r*T)*norm.cdf(-d2)
# # #             - S0 * np.exp(-q*T)*norm.cdf(-d1))

# # # def calc_lambda(r, q, sigma):
# # #     # λ = (r - q + σ²/2) / σ²
# # #     return (r - q + 0.5 * sigma**2) / (sigma**2)

# # # def calc_y(H, S0, K, T, sigma, r, q):
# # #     """
# # #     y = [ln(H^2/(S0*K)) / (sigma*sqrt(T))] + λ * sigma * sqrt(T)
# # #     """
# # #     lam = calc_lambda(r, q, sigma)
# # #     return (np.log((H**2)/(S0*K)) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

# # # def calc_x1(S0, H, T, sigma, r, q):
# # #     """
# # #     x1 = ln(S0/H)/(sigma*sqrt(T)) + λ*sigma*sqrt(T)
# # #     """
# # #     lam = calc_lambda(r, q, sigma)
# # #     return (np.log(S0/H) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

# # # def calc_y1(S0, H, T, sigma, r, q):
# # #     """
# # #     y1 = ln(H/S0)/(sigma*sqrt(T)) + λ*sigma*sqrt(T)
# # #     """
# # #     lam = calc_lambda(r, q, sigma)
# # #     return (np.log(H/S0) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

# # # # ------------------------------
# # # # 2) Main barrier pricing function
# # # # ------------------------------

# # # def barrier_option_price(S0, K, T, r, q, sigma, H, option_type):
# # #     """
# # #     Returns the price of a barrier option (various knock-in/out types).
# # #     Matches standard formulas from texts like Hull, with care to keep
# # #     exponents and sign conventions correct.
# # #     """
# # #     x1 = calc_x1(S0, H, T, sigma, r, q)
# # #     y1 = calc_y1(S0, H, T, sigma, r, q)
# # #     c = calc_c(S0, K, r, q, sigma, T)
# # #     p = calc_p(S0, K, r, q, sigma, T)
# # #     lam = calc_lambda(r, q, sigma)
# # #     y  = calc_y(H, S0, K, T, sigma, r, q)

# # #     # --------------------------------
# # #     # Down-and-in Call
# # #     # --------------------------------
# # #     if option_type == 'down-and-in call' and H <= K:
# # #         # cdi, for H <= K
# # #         cdi = (S0 * np.exp(-q*T) * (H/S0)**(2*lam) * norm.cdf(y)
# # #                - K * np.exp(-r*T) * (H/S0)**(2*lam - 2)
# # #                  * norm.cdf(y - sigma*np.sqrt(T)))
# # #         return cdi

# # #     elif option_type == 'down-and-in call' and H >= K:
# # #         # cdi = c - cdo. So we compute cdo from the standard expression
# # #         # cdo = ...
# # #         # Then cdi = c - cdo
# # #         term1 = S0*np.exp(-q*T)*norm.cdf(x1)
# # #         term2 = K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
# # #         term3 = S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(y1)
# # #         term4 = K*np.exp(-r*T)*(H/S0)**(2*lam - 2)*norm.cdf(y1 - sigma*np.sqrt(T))
# # #         cdo   = term1 - term2 - term3 + term4
# # #         if cdo < 0:
# # #             cdo = 0
# # #             cdi   = c - cdo
# # #             return cdi
# # #         else:
# # #             return cdi

# # #     # --------------------------------
# # #     # Down-and-out Call
# # #     # --------------------------------
# # #     elif option_type == 'down-and-out call' and H <= K:
# # #         # cdo = c - cdi
# # #         cdi = (S0 * np.exp(-q*T) * (H/S0)**(2*lam) * norm.cdf(y)
# # #                - K * np.exp(-r*T) * (H/S0)**(2*lam - 2)
# # #                  * norm.cdf(y - sigma*np.sqrt(T)))
# # #         cdo = c - cdi
# # #         return cdo

# # #     elif option_type == 'down-and-out call' and H >= K:
# # #         # This is the “If H > K” formula for the down-and-out call
# # #         term1 = S0 * np.exp(-q*T)*norm.cdf(x1)
# # #         term2 = K  * np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
# # #         term3 = S0 * np.exp(-q*T)*((H/S0)**(2*lam))*norm.cdf(y1)
# # #         term4 = K  * np.exp(-r*T)*((H/S0)**(2*lam - 2))*norm.cdf(y1 - sigma*np.sqrt(T))
# # #         cdo   = term1 - term2 - term3 + term4
        
# # #         if cdo < 0:
# # #             return 0
# # #         else:
# # #             return cdo

# # #     # --------------------------------
# # #     # Up-and-in Call
# # #     # --------------------------------
# # #     elif option_type == 'up-and-in call' and H > K:
# # #         # Standard up-and-in call for H > K
# # #         cui = (S0*np.exp(-q*T)*norm.cdf(x1)
# # #                - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
# # #                - S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(-y) - norm.cdf(-y1))
# # #                + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
# # #                  * (norm.cdf(-y + sigma*np.sqrt(T))
# # #                     - norm.cdf(-y1 + sigma*np.sqrt(T))))
# # #         return cui

# # #     elif option_type == 'up-and-in call' and H <= K:
# # #         # If barrier is below K, the up-and-in call is effectively the same as c
# # #         # or 0, depending on your setup.  Typically if H < S0 < K,
# # #         # the option knocks in only if S0 goes above H.  If you are sure
# # #         # you want to treat it as simply c, do so here:
# # #         return c

# # #     # --------------------------------
# # #     # Up-and-out Call
# # #     # --------------------------------
# # #     elif option_type == 'up-and-out call' and H <= K:
# # #         # If the barrier H <= K is below the current spot,
# # #         # often up-and-out call is worthless if it is truly "up" barrier?
# # #         return 0.0

# # #     elif option_type == 'up-and-out call' and H > K:
# # #         cui = (S0*np.exp(-q*T)*norm.cdf(x1)
# # #                - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
# # #                - S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(-y) - norm.cdf(-y1))
# # #                + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
# # #                  * (norm.cdf(-y + sigma*np.sqrt(T))
# # #                     - norm.cdf(-y1 + sigma*np.sqrt(T))))
# # #         cuo = c - cui
# # #         return cuo

# # #     # --------------------------------
# # #     # Up-and-in Put
# # #     # --------------------------------
# # #     elif option_type == 'up-and-in put' and H >= K:
# # #         pui = (-S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y)
# # #                + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
# # #                  * norm.cdf(-y + sigma*np.sqrt(T)))
# # #         return pui

# # #     elif option_type == 'up-and-in put' and H <= K:
# # #         # up-and-in put is the difference p - up-and-out put
# # #         # but for the simplified logic, we can just return p if the barrier is < K
# # #         return p

# # #     # --------------------------------
# # #     # Up-and-out Put
# # #     # --------------------------------
# # #     elif option_type == 'up-and-out put' and H >= K:
# # #         # puo = p - pui
# # #         pui = (-S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y)
# # #                + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
# # #                  * norm.cdf(-y + sigma*np.sqrt(T)))
# # #         puo = p - pui
# # #         return puo

# # #     elif option_type == 'up-and-out put' and H <= K:
# # #         # Standard formula for H <= K
# # #         puo = (
# # #             -S0*np.exp(-q*T)*norm.cdf(-x1)
# # #             + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
# # #             + S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y1)
# # #             - K*np.exp(-r*T)*(H/S0)**(2*lam - 2)*norm.cdf(-y1 + sigma*np.sqrt(T))
# # #         )
# # #         if puo < 0:
# # #             return 0
# # #         else:
# # #             return puo

# # #     # --------------------------------
# # #     # Down-and-in Put
# # #     # --------------------------------
# # #     elif option_type == 'down-and-in put' and H > K:
# # #         # If the barrier is above K, we often treat the down-and-in put as p
# # #         return p

# # #     elif option_type == 'down-and-in put' and H < K:
# # #         pdi = (
# # #             -S0*np.exp(-q*T)*norm.cdf(-x1)
# # #             + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
# # #             + S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(y) - norm.cdf(y1))
# # #             - K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
# # #               * (norm.cdf(y - sigma*np.sqrt(T))
# # #                  - norm.cdf(y1 - sigma*np.sqrt(T)))
# # #         )
# # #         return pdi

# # #     # --------------------------------
# # #     # Down-and-out Put
# # #     # --------------------------------
# # #     elif option_type == 'down-and-out put' and H > K:
# # #         # Typically worthless if H > K in certain setups
# # #         return 0.0

# # #     elif option_type == 'down-and-out put' and H < K:
# # #         pdi = (
# # #             -S0*np.exp(-q*T)*norm.cdf(-x1)
# # #             + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
# # #             + S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(y) - norm.cdf(y1))
# # #             - K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
# # #               * (norm.cdf(y - sigma*np.sqrt(T))
# # #                  - norm.cdf(y1 - sigma*np.sqrt(T)))
# # #         )
# # #         pdo = p - pdi
# # #         return pdo

# # #     # Fallback
# # #     return None

# # # # ------------------------------------------------------------------------------
# # # # Black–Scholes for vanilla (for CV)
# # # # ------------------------------------------------------------------------------
# # # def black_scholes_call(S0, K, T, r, sigma):
# # #     d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
# # #     d2 = d1 - sigma*np.sqrt(T)
# # #     return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# # # def black_scholes_put(S0, K, T, r, sigma):
# # #     d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
# # #     d2 = d1 - sigma*np.sqrt(T)
# # #     return K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)

# # # # ------------------------------------------------------------------------------
# # # # 3) Monte Carlo Approaches
# # # #    a) Plain
# # # #    b) Control Variate
# # # #    c) Antithetic
# # # # ------------------------------------------------------------------------------
# # # def mc_barrier_option_plain(S0, K, H, T, r, q, sigma, N, M, option_type):
# # #     """Plain MC for the chosen barrier type, discrete monitoring."""
# # #     dt   = T / M
# # #     disc = np.exp(-r*T)

# # #     is_call = 'call' in option_type.lower()
# # #     is_in   = 'in' in option_type.lower()
# # #     is_up   = 'up' in option_type.lower()

# # #     payoffs = np.zeros(N)
# # #     for i in range(N):
# # #         S_path = S0
# # #         barrier_breached = False

# # #         for _ in range(M):
# # #             z = np.random.randn()
# # #             S_path *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
# # #             # barrier check
# # #             if is_up and S_path >= H:
# # #                 barrier_breached = True
# # #             elif (not is_up) and S_path <= H:
# # #                 barrier_breached = True

# # #         # payoff
# # #         if is_in:
# # #             if barrier_breached:
# # #                 payoff = max(S_path-K, 0) if is_call else max(K-S_path, 0)
# # #             else:
# # #                 payoff = 0.0
# # #         else:
# # #             # knock-out
# # #             if barrier_breached:
# # #                 payoff = 0.0
# # #             else:
# # #                 payoff = max(S_path-K, 0) if is_call else max(K-S_path, 0)
# # #         payoffs[i] = payoff

# # #     return disc*np.mean(payoffs)

# # # def mc_barrier_option_cv(S0, K, H, T, r, q, sigma, N, M, option_type):
# # #     """Control Variate MC: same simulation as Plain, but adjust w/ vanilla call or put."""
# # #     dt   = T / M
# # #     disc = np.exp(-r*T)

# # #     is_call = 'call' in option_type.lower()
# # #     is_in   = 'in' in option_type.lower()
# # #     is_up   = 'up' in option_type.lower()

# # #     # pick the correct vanilla analytic
# # #     if is_call:
# # #         vanilla_analytic = black_scholes_call(S0, K, T, r, sigma)
# # #     else:
# # #         vanilla_analytic = black_scholes_put(S0, K, T, r, sigma)

# # #     barrier_payoffs = np.zeros(N)
# # #     vanilla_payoffs = np.zeros(N)

# # #     for i in range(N):
# # #         S_path = S0
# # #         barrier_breached = False

# # #         for _ in range(M):
# # #             z = np.random.randn()
# # #             S_path *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
# # #             if is_up and S_path >= H:
# # #                 barrier_breached = True
# # #             elif (not is_up) and S_path <= H:
# # #                 barrier_breached = True

# # #         if is_in:
# # #             if barrier_breached:
# # #                 payoff_barrier = max(S_path-K, 0) if is_call else max(K-S_path, 0)
# # #             else:
# # #                 payoff_barrier = 0.0
# # #         else:
# # #             if barrier_breached:
# # #                 payoff_barrier = 0.0
# # #             else:
# # #                 payoff_barrier = max(S_path-K, 0) if is_call else max(K-S_path, 0)
# # #         barrier_payoffs[i] = payoff_barrier

# # #         # vanilla payoff
# # #         if is_call:
# # #             payoff_vanilla = max(S_path-K, 0)
# # #         else:
# # #             payoff_vanilla = max(K-S_path, 0)
# # #         vanilla_payoffs[i] = payoff_vanilla

# # #     mc_barrier = disc*np.mean(barrier_payoffs)
# # #     mc_vanilla = disc*np.mean(vanilla_payoffs)

# # #     mc_cv = mc_barrier + (vanilla_analytic - mc_vanilla)
# # #     return mc_cv

# # # def mc_barrier_option_av(S0, K, H, T, r, q, sigma, N, M, option_type):
# # #     """
# # #     Antithetic Variates MC:
# # #     For each i in range(N), generate one pair of paths: (+z) and (-z).
# # #     Then average their payoffs.
# # #     """
# # #     dt   = T / M
# # #     disc = np.exp(-r*T)

# # #     is_call = 'call' in option_type.lower()
# # #     is_in   = 'in' in option_type.lower()
# # #     is_up   = 'up' in option_type.lower()

# # #     payoffs = np.zeros(N)

# # #     for i in range(N):
# # #         S_plus  = S0
# # #         S_minus = S0

# # #         breach_plus  = False
# # #         breach_minus = False

# # #         for _ in range(M):
# # #             z = np.random.randn()
# # #             # +z path
# # #             S_plus *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
# # #             if is_up and S_plus >= H:
# # #                 breach_plus = True
# # #             elif (not is_up) and S_plus <= H:
# # #                 breach_plus = True

# # #             # -z path
# # #             S_minus *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*(-z))
# # #             if is_up and S_minus >= H:
# # #                 breach_minus = True
# # #             elif (not is_up) and S_minus <= H:
# # #                 breach_minus = True

# # #         # payoff +z path
# # #         if is_in:
# # #             if breach_plus:
# # #                 payoff_plus = max(S_plus-K, 0) if is_call else max(K-S_plus, 0)
# # #             else:
# # #                 payoff_plus = 0.0
# # #         else:
# # #             # knock-out
# # #             if breach_plus:
# # #                 payoff_plus = 0.0
# # #             else:
# # #                 payoff_plus = max(S_plus-K, 0) if is_call else max(K-S_plus, 0)

# # #         # payoff -z path
# # #         if is_in:
# # #             if breach_minus:
# # #                 payoff_minus = max(S_minus-K, 0) if is_call else max(K-S_minus, 0)
# # #             else:
# # #                 payoff_minus = 0.0
# # #         else:
# # #             if breach_minus:
# # #                 payoff_minus = 0.0
# # #             else:
# # #                 payoff_minus = max(S_minus-K, 0) if is_call else max(K-S_minus, 0)

# # #         # average
# # #         payoffs[i] = 0.5*(payoff_plus + payoff_minus)

# # #     return disc*np.mean(payoffs)

# # # # ------------------------------------------------------------------------------
# # # # 4) Streamlit UI
# # # # ------------------------------------------------------------------------------

# # # st.set_page_config(page_title="Barrier Option Pricing: Plain, CV, Antithetic", layout="wide")

# # # # Sidebar
# # # st.sidebar.header("Input Parameters")
# # # option_type = st.sidebar.selectbox(
# # #     "Barrier Option Type",
# # #     [
# # #         "down-and-in call",
# # #         "down-and-out call",
# # #         "down-and-in put",
# # #         "down-and-out put",
# # #         "up-and-in call",
# # #         "up-and-out call",
# # #         "up-and-in put",
# # #         "up-and-out put",
# # #     ]
# # # )
# # # S0 = st.sidebar.number_input("Spot (S0)", value=100.0)
# # # K  = st.sidebar.number_input("Strike (K)", value=100.0)
# # # H  = st.sidebar.number_input("Barrier (H)", value=80.0)
# # # T  = st.sidebar.number_input("Maturity (T)", value=1.0)
# # # r  = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05)
# # # q  = st.sidebar.number_input("Dividend Yield (q)", value=0.03)
# # # sigma = st.sidebar.number_input("Volatility (σ)", value=0.2)
# # # rebate = st.sidebar.number_input("Rebate (unused)", value=0.0)

# # # st.sidebar.write("MC Steps (M) and # of Paths (N list)")
# # # M = st.sidebar.number_input("Time Steps (M)", min_value=1, value=50)
# # # default_N = "1000,2000,5000"
# # # N_str = st.sidebar.text_input("Comma-separated N values:", default_N)
# # # N_list = []
# # # try:
# # #     N_list = sorted([int(x.strip()) for x in N_str.split(",") if int(x.strip())>0])
# # # except:
# # #     pass

# # # # Tabs
# # # tab1, tab2 = st.tabs(["Closed-Form", "MC Analysis"])

# # # # ------------------------------------------------------------------------------
# # # # Tab 1: Closed-Form
# # # # ------------------------------------------------------------------------------
# # # with tab1:
# # #     st.title("Closed-Form for Selected Barrier")
# # #     cf_price = barrier_option_price(S0, K, T, r, q, sigma, H, option_type)
# # #     if cf_price is not None:
# # #         st.subheader(f"{option_type.capitalize()} Price = {cf_price:.4f}")
# # #     else:
# # #         st.write("No closed-form formula implemented or invalid parameters.")

# # # # ------------------------------------------------------------------------------
# # # # Tab 2: Monte Carlo
# # # # ------------------------------------------------------------------------------
# # # with tab2:
# # #     st.title("Monte Carlo: Plain, Control Variate, Antithetic")

# # #     if st.button("Run MC"):
# # #         # We'll do three separate estimates:
# # #         # 1) Plain
# # #         # 2) CV
# # #         # 3) Antithetic
# # #         # for each N in N_list

# # #         # Also get closed-form if available
# # #         closed_form = barrier_option_price(S0, K, T, r, q, sigma, H, option_type)

# # #         # We'll build a DataFrame with columns: N, Plain, CV, AV
# # #         results_plain = []
# # #         results_cv    = []
# # #         results_av    = []

# # #         for N in N_list:
# # #             # Plain
# # #             mc_plain = mc_barrier_option_plain(S0, K, H, T, r, q, sigma, N, M, option_type)

# # #             # CV
# # #             mc_cv = mc_barrier_option_cv(S0, K, H, T, r, q, sigma, N, M, option_type)

# # #             # Antithetic
# # #             mc_av = mc_barrier_option_av(S0, K, H, T, r, q, sigma, N, M, option_type)

# # #             results_plain.append(mc_plain)
# # #             results_cv.append(mc_cv)
# # #             results_av.append(mc_av)

# # #         df = pd.DataFrame({
# # #             "N": N_list,
# # #             "Plain": results_plain,
# # #             "CV": results_cv,
# # #             "AV": results_av
# # #         })

# # #         st.subheader("1) MC Plain vs. CV vs. Analytical")
# # #         fig1 = px.line(
# # #             df, x="N", y=["Plain", "CV"], markers=True,
# # #             title=f"{option_type.capitalize()} - Plain vs. CV"
# # #         )
# # #         if closed_form is not None:
# # #             fig1.add_hline(
# # #                 y=closed_form, line_dash="dash", line_color="red",
# # #                 annotation_text=f"Analytical={closed_form:.4f}"
# # #             )
# # #         st.plotly_chart(fig1, use_container_width=True)

# # #         st.subheader("2) MC Plain vs. Antithetic vs. Analytical")
# # #         fig2 = px.line(
# # #             df, x="N", y=["Plain", "AV"], markers=True,
# # #             title=f"{option_type.capitalize()} - Plain vs. Antithetic"
# # #         )
# # #         if closed_form is not None:
# # #             fig2.add_hline(
# # #                 y=closed_form, line_dash="dash", line_color="red",
# # #                 annotation_text=f"Analytical={closed_form:.4f}"
# # #             )
# # #         st.plotly_chart(fig2, use_container_width=True)

# # #         st.write("### Full Table of Results")
# # #         st.dataframe(df.style.format("{:.4f}"))

# # #         if closed_form is not None:
# # #             st.write(f"**Closed-Form** = {closed_form:.4f}")

# # #     else:
# # #         st.info("Click 'Run MC' to see Plain, CV, and Antithetic results.")

# # import streamlit as st
# # import numpy as np
# # import pandas as pd
# # import plotly.express as px
# # from scipy.stats import norm

# # # ------------------------------
# # # 1) Helper functions
# # # ------------------------------

# # def calc_d1(S0, K, r, q, sigma, T):
# #     return (np.log(S0 / K) + (r - q + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))

# # def calc_d2(S0, K, r, q, sigma, T):
# #     return calc_d1(S0, K, r, q, sigma, T) - sigma * np.sqrt(T)

# # def calc_c(S0, K, r, q, sigma, T):
# #     d1 = calc_d1(S0, K, r, q, sigma, T)
# #     d2 = calc_d2(S0, K, r, q, sigma, T)
# #     return (S0 * np.exp(-q*T)*norm.cdf(d1)
# #             - K * np.exp(-r*T)*norm.cdf(d2))

# # def calc_p(S0, K, r, q, sigma, T):
# #     d1 = calc_d1(S0, K, r, q, sigma, T)
# #     d2 = calc_d2(S0, K, r, q, sigma, T)
# #     return (K * np.exp(-r*T)*norm.cdf(-d2)
# #             - S0 * np.exp(-q*T)*norm.cdf(-d1))

# # def calc_lambda(r, q, sigma):
# #     # λ = (r - q + σ²/2) / σ²
# #     return (r - q + 0.5 * sigma**2) / (sigma**2)

# # def calc_y(H, S0, K, T, sigma, r, q):
# #     """
# #     y = [ln(H^2/(S0*K)) / (sigma*sqrt(T))] + λ * sigma * sqrt(T)
# #     """
# #     lam = calc_lambda(r, q, sigma)
# #     return (np.log((H**2)/(S0*K)) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

# # def calc_x1(S0, H, T, sigma, r, q):
# #     """
# #     x1 = ln(S0/H)/(sigma*sqrt(T)) + λ*sigma*sqrt(T)
# #     """
# #     lam = calc_lambda(r, q, sigma)
# #     return (np.log(S0/H) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

# # def calc_y1(S0, H, T, sigma, r, q):
# #     """
# #     y1 = ln(H/S0)/(sigma*sqrt(T)) + λ*sigma*sqrt(T)
# #     """
# #     lam = calc_lambda(r, q, sigma)
# #     return (np.log(H/S0) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

# # # ------------------------------
# # # 2) Main barrier pricing function (closed-form)
# # # ------------------------------

# # def barrier_option_price(S0, K, T, r, q, sigma, H, option_type):
# #     """
# #     Returns the price of a barrier option (various knock-in/out types).
# #     Matches standard formulas from texts like Hull.
# #     """
# #     x1 = calc_x1(S0, H, T, sigma, r, q)
# #     y1 = calc_y1(S0, H, T, sigma, r, q)
# #     c = calc_c(S0, K, r, q, sigma, T)
# #     p = calc_p(S0, K, r, q, sigma, T)
# #     lam = calc_lambda(r, q, sigma)
# #     y  = calc_y(H, S0, K, T, sigma, r, q)

# #     # The code below uses standard “in” vs. “out” formulas, separated by barrier < or > relationships.
# #     # For brevity, these are the same logic you provided earlier.

# #     # down-and-in call, down-and-out call, up-and-in call, up-and-out call, etc.
# #     # ...
# #     # (Same as your original snippet.)

# #     # --------------------------------
# #     # Down-and-in Call
# #     # --------------------------------
# #     if option_type == 'down-and-in call' and H <= K:
# #         cdi = (S0 * np.exp(-q*T) * (H/S0)**(2*lam) * norm.cdf(y)
# #                - K * np.exp(-r*T) * (H/S0)**(2*lam - 2)
# #                  * norm.cdf(y - sigma*np.sqrt(T)))
# #         return cdi

# #     elif option_type == 'down-and-in call' and H >= K:
# #         term1 = S0*np.exp(-q*T)*norm.cdf(x1)
# #         term2 = K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
# #         term3 = S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(y1)
# #         term4 = K*np.exp(-r*T)*(H/S0)**(2*lam - 2)*norm.cdf(y1 - sigma*np.sqrt(T))
# #         cdo   = term1 - term2 - term3 + term4
# #         cdo   = max(cdo, 0.0)
# #         cdi   = c - cdo
# #         return cdi

# #     # --------------------------------
# #     # Down-and-out Call
# #     # --------------------------------
# #     elif option_type == 'down-and-out call' and H <= K:
# #         cdi = (S0 * np.exp(-q*T) * (H/S0)**(2*lam) * norm.cdf(y)
# #                - K * np.exp(-r*T) * (H/S0)**(2*lam - 2)
# #                  * norm.cdf(y - sigma*np.sqrt(T)))
# #         cdo = c - cdi
# #         return cdo

# #     elif option_type == 'down-and-out call' and H >= K:
# #         term1 = S0 * np.exp(-q*T)*norm.cdf(x1)
# #         term2 = K  * np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
# #         term3 = S0 * np.exp(-q*T)*((H/S0)**(2*lam))*norm.cdf(y1)
# #         term4 = K  * np.exp(-r*T)*((H/S0)**(2*lam - 2))*norm.cdf(y1 - sigma*np.sqrt(T))
# #         cdo   = term1 - term2 - term3 + term4
# #         return max(cdo, 0.0)

# #     # --------------------------------
# #     # Up-and-in Call
# #     # --------------------------------
# #     elif option_type == 'up-and-in call' and H > K:
# #         cui = (S0*np.exp(-q*T)*norm.cdf(x1)
# #                - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
# #                - S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(-y) - norm.cdf(-y1))
# #                + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
# #                  * (norm.cdf(-y + sigma*np.sqrt(T))
# #                     - norm.cdf(-y1 + sigma*np.sqrt(T))))
# #         return cui

# #     elif option_type == 'up-and-in call' and H <= K:
# #         return c

# #     # --------------------------------
# #     # Up-and-out Call
# #     # --------------------------------
# #     elif option_type == 'up-and-out call' and H <= K:
# #         return 0.0

# #     elif option_type == 'up-and-out call' and H > K:
# #         cui = (S0*np.exp(-q*T)*norm.cdf(x1)
# #                - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
# #                - S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(-y) - norm.cdf(-y1))
# #                + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
# #                  * (norm.cdf(-y + sigma*np.sqrt(T))
# #                     - norm.cdf(-y1 + sigma*np.sqrt(T))))
# #         return c - cui

# #     # --------------------------------
# #     # Up-and-in Put
# #     # --------------------------------
# #     elif option_type == 'up-and-in put' and H >= K:
# #         pui = (-S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y)
# #                + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
# #                  * norm.cdf(-y + sigma*np.sqrt(T)))
# #         return pui

# #     elif option_type == 'up-and-in put' and H <= K:
# #         return p

# #     # --------------------------------
# #     # Up-and-out Put
# #     # --------------------------------
# #     elif option_type == 'up-and-out put' and H >= K:
# #         pui = (-S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y)
# #                + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
# #                  * norm.cdf(-y + sigma*np.sqrt(T)))
# #         puo = p - pui
# #         return puo

# #     elif option_type == 'up-and-out put' and H <= K:
# #         puo = (
# #             -S0*np.exp(-q*T)*norm.cdf(-x1)
# #             + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
# #             + S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y1)
# #             - K*np.exp(-r*T)*(H/S0)**(2*lam - 2)*norm.cdf(-y1 + sigma*np.sqrt(T))
# #         )
# #         return max(puo, 0.0)

# #     # --------------------------------
# #     # Down-and-in Put
# #     # --------------------------------
# #     elif option_type == 'down-and-in put' and H > K:
# #         return p

# #     elif option_type == 'down-and-in put' and H < K:
# #         pdi = (
# #             -S0*np.exp(-q*T)*norm.cdf(-x1)
# #             + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
# #             + S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(y) - norm.cdf(y1))
# #             - K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
# #               * (norm.cdf(y - sigma*np.sqrt(T))
# #                  - norm.cdf(y1 - sigma*np.sqrt(T)))
# #         )
# #         return pdi

# #     # --------------------------------
# #     # Down-and-out Put
# #     # --------------------------------
# #     elif option_type == 'down-and-out put' and H > K:
# #         return 0.0

# #     elif option_type == 'down-and-out put' and H < K:
# #         pdi = (
# #             -S0*np.exp(-q*T)*norm.cdf(-x1)
# #             + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
# #             + S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(y) - norm.cdf(y1))
# #             - K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
# #               * (norm.cdf(y - sigma*np.sqrt(T))
# #                  - norm.cdf(y1 - sigma*np.sqrt(T)))
# #         )
# #         pdo = p - pdi
# #         return pdo

# #     # Fallback
# #     return None

# # # ------------------------------------------------------------------------------
# # # Black–Scholes for vanilla (for CV)
# # # ------------------------------------------------------------------------------
# # def black_scholes_call(S0, K, T, r, sigma):
# #     d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
# #     d2 = d1 - sigma*np.sqrt(T)
# #     return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# # def black_scholes_put(S0, K, T, r, sigma):
# #     d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
# #     d2 = d1 - sigma*np.sqrt(T)
# #     return K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)

# # # ------------------------------------------------------------------------------
# # # 3) Monte Carlo Approaches:
# # #    a) Plain
# # #    b) Control Variate (CV)
# # #    c) Antithetic Variates (AV)
# # #    d) Combined (CV + AV)
# # # ------------------------------------------------------------------------------

# # def mc_barrier_option_plain(S0, K, H, T, r, q, sigma, N, M, option_type):
# #     """Plain MC for the chosen barrier type, discrete monitoring."""
# #     #remove seeds to produce other results
# #     np.random.seed(42)
# #     dt   = T / M
# #     disc = np.exp(-r*T)

# #     is_call = 'call' in option_type.lower()
# #     is_in   = 'in' in option_type.lower()
# #     is_up   = 'up' in option_type.lower()

# #     payoffs = np.zeros(N)
# #     for i in range(N):
# #         S_path = S0
# #         barrier_breached = False

# #         for _ in range(M):
# #             z = np.random.randn()
# #             S_path *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
# #             # barrier check
# #             if is_up and S_path >= H:
# #                 barrier_breached = True
# #             elif (not is_up) and S_path <= H:
# #                 barrier_breached = True

# #         # payoff
# #         if is_in:
# #             if barrier_breached:
# #                 payoff = max(S_path-K, 0) if is_call else max(K-S_path, 0)
# #             else:
# #                 payoff = 0.0
# #         else:
# #             # knock-out
# #             if barrier_breached:
# #                 payoff = 0.0
# #             else:
# #                 payoff = max(S_path-K, 0) if is_call else max(K-S_path, 0)
# #         payoffs[i] = payoff

# #     return disc*np.mean(payoffs)

# # def mc_barrier_option_cv(S0, K, H, T, r, q, sigma, N, M, option_type):
# #     """Control Variate MC: uses the same paths as Plain, but adjust w/ vanilla."""
# #     np.random.seed(42)
# #     dt   = T / M
# #     disc = np.exp(-r*T)

# #     is_call = 'call' in option_type.lower()
# #     is_in   = 'in' in option_type.lower()
# #     is_up   = 'up' in option_type.lower()

# #     # pick the correct vanilla analytic
# #     if is_call:
# #         vanilla_analytic = black_scholes_call(S0, K, T, r, sigma)
# #     else:
# #         vanilla_analytic = black_scholes_put(S0, K, T, r, sigma)

# #     barrier_payoffs = np.zeros(N)
# #     vanilla_payoffs = np.zeros(N)

# #     for i in range(N):
# #         S_path = S0
# #         barrier_breached = False

# #         for _ in range(M):
# #             z = np.random.randn()
# #             S_path *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
# #             if is_up and S_path >= H:
# #                 barrier_breached = True
# #             elif (not is_up) and S_path <= H:
# #                 barrier_breached = True

# #         if is_in:
# #             if barrier_breached:
# #                 payoff_barrier = max(S_path-K, 0) if is_call else max(K-S_path, 0)
# #             else:
# #                 payoff_barrier = 0.0
# #         else:
# #             if barrier_breached:
# #                 payoff_barrier = 0.0
# #             else:
# #                 payoff_barrier = max(S_path-K, 0) if is_call else max(K-S_path, 0)
# #         barrier_payoffs[i] = payoff_barrier

# #         # vanilla payoff
# #         if is_call:
# #             payoff_vanilla = max(S_path-K, 0)
# #         else:
# #             payoff_vanilla = max(K-S_path, 0)
# #         vanilla_payoffs[i] = payoff_vanilla

# #     mc_barrier = disc*np.mean(barrier_payoffs)
# #     mc_vanilla = disc*np.mean(vanilla_payoffs)

# #     mc_cv = mc_barrier + (vanilla_analytic - mc_vanilla)
# #     return mc_cv

# # def mc_barrier_option_av(S0, K, H, T, r, q, sigma, N, M, option_type):
# #     """Antithetic Variates MC: generate paired (+z, -z) paths, average their payoffs."""
# #     np.random.seed(42)
# #     dt   = T / M
# #     disc = np.exp(-r*T)

# #     is_call = 'call' in option_type.lower()
# #     is_in   = 'in' in option_type.lower()
# #     is_up   = 'up' in option_type.lower()

# #     payoffs = np.zeros(N)

# #     for i in range(N):
# #         S_plus  = S0
# #         S_minus = S0

# #         breach_plus  = False
# #         breach_minus = False

# #         for _ in range(M):
# #             z = np.random.randn()
# #             # +z path
# #             S_plus *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
# #             if is_up and S_plus >= H:
# #                 breach_plus = True
# #             elif (not is_up) and S_plus <= H:
# #                 breach_plus = True

# #             # -z path
# #             S_minus *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*(-z))
# #             if is_up and S_minus >= H:
# #                 breach_minus = True
# #             elif (not is_up) and S_minus <= H:
# #                 breach_minus = True

# #         # payoff +z path
# #         if is_in:
# #             if breach_plus:
# #                 payoff_plus = max(S_plus-K, 0) if is_call else max(K-S_plus, 0)
# #             else:
# #                 payoff_plus = 0.0
# #         else:
# #             if breach_plus:
# #                 payoff_plus = 0.0
# #             else:
# #                 payoff_plus = max(S_plus-K, 0) if is_call else max(K-S_plus, 0)

# #         # payoff -z path
# #         if is_in:
# #             if breach_minus:
# #                 payoff_minus = max(S_minus-K, 0) if is_call else max(K-S_minus, 0)
# #             else:
# #                 payoff_minus = 0.0
# #         else:
# #             if breach_minus:
# #                 payoff_minus = 0.0
# #             else:
# #                 payoff_minus = max(S_minus-K, 0) if is_call else max(K-S_minus, 0)

# #         payoffs[i] = 0.5*(payoff_plus + payoff_minus)

# #     return disc*np.mean(payoffs)

# # def mc_barrier_option_combined(S0, K, H, T, r, q, sigma, N, M, option_type):
# #     """
# #     Combined Approach: Antithetic Variates + Control Variate.
# #     We generate paired paths, compute the barrier option payoff,
# #     and also compute the vanilla payoff (from the +z path) for CV.
# #     """
# #     np.random.seed(42)
# #     dt   = T / M
# #     disc = np.exp(-r*T)

# #     is_call = 'call' in option_type.lower()
# #     is_in   = 'in' in option_type.lower()
# #     is_up   = 'up' in option_type.lower()

# #     # For the control variate correction
# #     if is_call:
# #         vanilla_analytic = black_scholes_call(S0, K, T, r, sigma)
# #     else:
# #         vanilla_analytic = black_scholes_put(S0, K, T, r, sigma)

# #     barrier_payoffs = np.zeros(N)
# #     vanilla_payoffs = np.zeros(N)

# #     for i in range(N):
# #         # +z, -z paths
# #         S_plus  = S0
# #         S_minus = S0
# #         breach_plus  = False
# #         breach_minus = False

# #         for _ in range(M):
# #             z = np.random.randn()
# #             # +z path
# #             S_plus *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
# #             if is_up and S_plus >= H:
# #                 breach_plus = True
# #             elif (not is_up) and S_plus <= H:
# #                 breach_plus = True

# #             # -z path
# #             S_minus *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*(-z))
# #             if is_up and S_minus >= H:
# #                 breach_minus = True
# #             elif (not is_up) and S_minus <= H:
# #                 breach_minus = True

# #         # Payoff for barrier option on +z path
# #         if is_in:
# #             if breach_plus:
# #                 payoff_plus = max(S_plus - K, 0) if is_call else max(K - S_plus, 0)
# #             else:
# #                 payoff_plus = 0.0
# #         else:
# #             # knock-out
# #             if breach_plus:
# #                 payoff_plus = 0.0
# #             else:
# #                 payoff_plus = max(S_plus - K, 0) if is_call else max(K - S_plus, 0)

# #         # Payoff for barrier option on -z path
# #         if is_in:
# #             if breach_minus:
# #                 payoff_minus = max(S_minus - K, 0) if is_call else max(K - S_minus, 0)
# #             else:
# #                 payoff_minus = 0.0
# #         else:
# #             if breach_minus:
# #                 payoff_minus = 0.0
# #             else:
# #                 payoff_minus = max(S_minus - K, 0) if is_call else max(K - S_minus, 0)

# #         # Antithetic average for the barrier option payoff
# #         barrier_payoff = 0.5*(payoff_plus + payoff_minus)

# #         # For the control variate, we only need the +z path's vanilla payoff
# #         if is_call:
# #             payoff_vanilla = max(S_plus - K, 0)
# #         else:
# #             payoff_vanilla = max(K - S_plus, 0)

# #         barrier_payoffs[i] = barrier_payoff
# #         vanilla_payoffs[i] = payoff_vanilla

# #     barrier_mc = disc * np.mean(barrier_payoffs)
# #     vanilla_mc = disc * np.mean(vanilla_payoffs)
# #     # Control variate adjustment
# #     barrier_combined = barrier_mc + (vanilla_analytic - vanilla_mc)
# #     return barrier_combined

# # # ------------------------------------------------------------------------------
# # # 4) Streamlit UI
# # # ------------------------------------------------------------------------------

# # st.set_page_config(page_title="Barrier Option Pricing: Plain, CV, AV, Combined")

# # # Sidebar
# # st.sidebar.header("Input Parameters")
# # option_type = st.sidebar.selectbox(
# #     "Barrier Option Type",
# #     [
# #         "down-and-in call",
# #         "down-and-out call",
# #         "down-and-in put",
# #         "down-and-out put",
# #         "up-and-in call",
# #         "up-and-out call",
# #         "up-and-in put",
# #         "up-and-out put",
# #     ]
# # )
# # S0 = st.sidebar.number_input("Spot (S0)", value=100.0)
# # K  = st.sidebar.number_input("Strike (K)", value=100.0)
# # H  = st.sidebar.number_input("Barrier (H)", value=80.0)
# # T  = st.sidebar.number_input("Maturity (T)", value=1.0)
# # r  = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05)
# # q  = st.sidebar.number_input("Dividend Yield (q)", value=0.03)
# # sigma = st.sidebar.number_input("Volatility (σ)", value=0.2)
# # rebate = st.sidebar.number_input("Rebate (unused)", value=0.0)

# # st.sidebar.write("MC Steps (M) and # of Paths (N list)")
# # M = st.sidebar.number_input("Time Steps (M)", min_value=1, value=50)
# # default_N = "1000,2000,5000"
# # N_str = st.sidebar.text_input("Comma-separated N values:", default_N)
# # N_list = []
# # try:
# #     N_list = sorted([int(x.strip()) for x in N_str.split(",") if int(x.strip())>0])
# # except:
# #     pass

# # # Tabs
# # tab1, tab2 = st.tabs(["Closed-Form", "MC Analysis"])

# # # ------------------------------------------------------------------------------
# # # Tab 1: Closed-Form
# # # ------------------------------------------------------------------------------
# # with tab1:
# #     st.title("Closed-Form for Selected Barrier")
# #     cf_price = barrier_option_price(S0, K, T, r, q, sigma, H, option_type)
# #     if cf_price is not None:
# #         st.subheader(f"{option_type.capitalize()} Price = {cf_price:.4f}")
# #     else:
# #         st.write("No closed-form formula implemented or invalid parameters.")

# # # ------------------------------------------------------------------------------
# # # Tab 2: Monte Carlo
# # # ------------------------------------------------------------------------------
# # with tab2:
# #     st.title("Monte Carlo: Plain, CV, Antithetic, Combined")

# #     if st.button("Run MC"):
# #         closed_form = barrier_option_price(S0, K, T, r, q, sigma, H, option_type)

# #         # We'll build a DataFrame with columns: N, Plain, CV, AV, Combined
# #         results_plain = []
# #         results_cv    = []
# #         results_av    = []
# #         results_comb  = []

# #         for N in N_list:
# #             # 1) Plain
# #             mc_plain = mc_barrier_option_plain(S0, K, H, T, r, q, sigma, N, M, option_type)
# #             # 2) CV
# #             mc_cv = mc_barrier_option_cv(S0, K, H, T, r, q, sigma, N, M, option_type)
# #             # 3) AV
# #             mc_av = mc_barrier_option_av(S0, K, H, T, r, q, sigma, N, M, option_type)
# #             # 4) Combined
# #             mc_comb = mc_barrier_option_combined(S0, K, H, T, r, q, sigma, N, M, option_type)

# #             results_plain.append(mc_plain)
# #             results_cv.append(mc_cv)
# #             results_av.append(mc_av)
# #             results_comb.append(mc_comb)

# #         df = pd.DataFrame({
# #             "N": N_list,
# #             "Plain": results_plain,
# #             "CV": results_cv,
# #             "AV": results_av,
# #             "Combined": results_comb
# #         })

# #         st.subheader("1) MC: Plain vs. CV vs. Analytical")
# #         fig1 = px.line(
# #             df, x="N", y=["Plain", "CV"], markers=True,
# #             title=f"{option_type.capitalize()} - Plain vs. CV"
# #         )
# #         if closed_form is not None:
# #             fig1.add_hline(
# #                 y=closed_form, line_dash="dash", line_color="red",
# #                 annotation_text=f"Analytical={closed_form:.4f}"
# #             )
# #         st.plotly_chart(fig1, use_container_width=True)

# #         st.subheader("2) MC: Plain vs. AV vs. Analytical")
# #         fig2 = px.line(
# #             df, x="N", y=["Plain", "AV"], markers=True,
# #             title=f"{option_type.capitalize()} - Plain vs. Antithetic"
# #         )
# #         if closed_form is not None:
# #             fig2.add_hline(
# #                 y=closed_form, line_dash="dash", line_color="red",
# #                 annotation_text=f"Analytical={closed_form:.4f}"
# #             )
# #         st.plotly_chart(fig2, use_container_width=True)

# #         st.subheader("3) MC: Plain vs. Combined vs. Analytical")
# #         fig3 = px.line(
# #             df, x="N", y=["Plain", "Combined"], markers=True,
# #             title=f"{option_type.capitalize()} - Plain vs. Combined (CV+AV)"
# #         )
# #         if closed_form is not None:
# #             fig3.add_hline(
# #                 y=closed_form, line_dash="dash", line_color="red",
# #                 annotation_text=f"Analytical={closed_form:.4f}"
# #             )
# #         st.plotly_chart(fig3, use_container_width=True)

# #         st.subheader("4) Full Table of Results")
# #         st.dataframe(df.style.format("{:.4f}"))

# #         if closed_form is not None:
# #             st.write(f"**Closed-Form** = {closed_form:.4f}")

# #     else:
# #         st.info("Click 'Run MC' to see Plain, CV, AV, and Combined results.")

# import streamlit as st
# import numpy as np
# import pandas as pd
# import plotly.express as px
# from scipy.stats import norm

# # ------------------------------
# # 1) Helper functions
# # ------------------------------

# def calc_d1(S0, K, r, q, sigma, T):
#     return (np.log(S0 / K) + (r - q + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))

# def calc_d2(S0, K, r, q, sigma, T):
#     return calc_d1(S0, K, r, q, sigma, T) - sigma*np.sqrt(T)

# def calc_c(S0, K, r, q, sigma, T):
#     d1 = calc_d1(S0, K, r, q, sigma, T)
#     d2 = calc_d2(S0, K, r, q, sigma, T)
#     return (S0 * np.exp(-q*T)*norm.cdf(d1)
#             - K * np.exp(-r*T)*norm.cdf(d2))

# def calc_p(S0, K, r, q, sigma, T):
#     d1 = calc_d1(S0, K, r, q, sigma, T)
#     d2 = calc_d2(S0, K, r, q, sigma, T)
#     return (K*np.exp(-r*T)*norm.cdf(-d2)
#             - S0*np.exp(-q*T)*norm.cdf(-d1))

# def calc_lambda(r, q, sigma):
#     return (r - q + 0.5*sigma**2) / (sigma**2)

# def calc_y(H, S0, K, T, sigma, r, q):
#     lam = calc_lambda(r, q, sigma)
#     return (np.log((H**2)/(S0*K)) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

# def calc_x1(S0, H, T, sigma, r, q):
#     lam = calc_lambda(r, q, sigma)
#     return (np.log(S0/H) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

# def calc_y1(S0, H, T, sigma, r, q):
#     lam = calc_lambda(r, q, sigma)
#     return (np.log(H/S0) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

# # ------------------------------
# # 2) Barrier Option Closed-Form
# # ------------------------------

# def barrier_option_price(S0, K, T, r, q, sigma, H, option_type):
#     """
#     Returns the price of a barrier option (various knock-in/out types).
#     Matches standard formulas from texts like Hull.
#     """
#     x1 = calc_x1(S0, H, T, sigma, r, q)
#     y1 = calc_y1(S0, H, T, sigma, r, q)
#     c = calc_c(S0, K, r, q, sigma, T)
#     p = calc_p(S0, K, r, q, sigma, T)
#     lam = calc_lambda(r, q, sigma)
#     y  = calc_y(H, S0, K, T, sigma, r, q)

#     # The code below uses standard “in” vs. “out” formulas, separated by barrier < or > relationships.
#     # For brevity, these are the same logic you provided earlier.

#     # down-and-in call, down-and-out call, up-and-in call, up-and-out call, etc.
#     # ...
#     # (Same as your original snippet.)

#     # --------------------------------
#     # Down-and-in Call
#     # --------------------------------
#     if option_type == 'down-and-in call' and H <= K:
#         cdi = (S0 * np.exp(-q*T) * (H/S0)**(2*lam) * norm.cdf(y)
#                - K * np.exp(-r*T) * (H/S0)**(2*lam - 2)
#                  * norm.cdf(y - sigma*np.sqrt(T)))
#         return cdi

#     elif option_type == 'down-and-in call' and H >= K:
#         term1 = S0*np.exp(-q*T)*norm.cdf(x1)
#         term2 = K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
#         term3 = S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(y1)
#         term4 = K*np.exp(-r*T)*(H/S0)**(2*lam - 2)*norm.cdf(y1 - sigma*np.sqrt(T))
#         cdo   = term1 - term2 - term3 + term4
#         cdo   = max(cdo, 0.0)
#         cdi   = c - cdo
#         return cdi

#     # --------------------------------
#     # Down-and-out Call
#     # --------------------------------
#     elif option_type == 'down-and-out call' and H <= K:
#         cdi = (S0 * np.exp(-q*T) * (H/S0)**(2*lam) * norm.cdf(y)
#                - K * np.exp(-r*T) * (H/S0)**(2*lam - 2)
#                  * norm.cdf(y - sigma*np.sqrt(T)))
#         cdo = c - cdi
#         return cdo

#     elif option_type == 'down-and-out call' and H >= K:
#         term1 = S0 * np.exp(-q*T)*norm.cdf(x1)
#         term2 = K  * np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
#         term3 = S0 * np.exp(-q*T)*((H/S0)**(2*lam))*norm.cdf(y1)
#         term4 = K  * np.exp(-r*T)*((H/S0)**(2*lam - 2))*norm.cdf(y1 - sigma*np.sqrt(T))
#         cdo   = term1 - term2 - term3 + term4
#         return max(cdo, 0.0)

#     # --------------------------------
#     # Up-and-in Call
#     # --------------------------------
#     elif option_type == 'up-and-in call' and H > K:
#         cui = (S0*np.exp(-q*T)*norm.cdf(x1)
#                - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
#                - S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(-y) - norm.cdf(-y1))
#                + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
#                  * (norm.cdf(-y + sigma*np.sqrt(T))
#                     - norm.cdf(-y1 + sigma*np.sqrt(T))))
#         return cui

#     elif option_type == 'up-and-in call' and H <= K:
#         return c

#     # --------------------------------
#     # Up-and-out Call
#     # --------------------------------
#     elif option_type == 'up-and-out call' and H <= K:
#         return 0.0

#     elif option_type == 'up-and-out call' and H > K:
#         cui = (S0*np.exp(-q*T)*norm.cdf(x1)
#                - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
#                - S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(-y) - norm.cdf(-y1))
#                + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
#                  * (norm.cdf(-y + sigma*np.sqrt(T))
#                     - norm.cdf(-y1 + sigma*np.sqrt(T))))
#         return c - cui

#     # --------------------------------
#     # Up-and-in Put
#     # --------------------------------
#     elif option_type == 'up-and-in put' and H >= K:
#         pui = (-S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y)
#                + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
#                  * norm.cdf(-y + sigma*np.sqrt(T)))
#         return pui

#     elif option_type == 'up-and-in put' and H <= K:
#         return p

#     # --------------------------------
#     # Up-and-out Put
#     # --------------------------------
#     elif option_type == 'up-and-out put' and H >= K:
#         pui = (-S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y)
#                + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
#                  * norm.cdf(-y + sigma*np.sqrt(T)))
#         puo = p - pui
#         return puo

#     elif option_type == 'up-and-out put' and H <= K:
#         puo = (
#             -S0*np.exp(-q*T)*norm.cdf(-x1)
#             + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
#             + S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y1)
#             - K*np.exp(-r*T)*(H/S0)**(2*lam - 2)*norm.cdf(-y1 + sigma*np.sqrt(T))
#         )
#         return max(puo, 0.0)

#     # --------------------------------
#     # Down-and-in Put
#     # --------------------------------
#     elif option_type == 'down-and-in put' and H > K:
#         return p

#     elif option_type == 'down-and-in put' and H < K:
#         pdi = (
#             -S0*np.exp(-q*T)*norm.cdf(-x1)
#             + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
#             + S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(y) - norm.cdf(y1))
#             - K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
#               * (norm.cdf(y - sigma*np.sqrt(T))
#                  - norm.cdf(y1 - sigma*np.sqrt(T)))
#         )
#         return pdi

#     # --------------------------------
#     # Down-and-out Put
#     # --------------------------------
#     elif option_type == 'down-and-out put' and H > K:
#         return 0.0

#     elif option_type == 'down-and-out put' and H < K:
#         pdi = (
#             -S0*np.exp(-q*T)*norm.cdf(-x1)
#             + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
#             + S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(y) - norm.cdf(y1))
#             - K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
#               * (norm.cdf(y - sigma*np.sqrt(T))
#                  - norm.cdf(y1 - sigma*np.sqrt(T)))
#         )
#         pdo = p - pdi
#         return pdo

#     # Fallback
#     return None
# # ------------------------------------------------------------------------------
# # Black–Scholes for vanilla (for control variate)
# # ------------------------------------------------------------------------------
# def black_scholes_call(S0, K, T, r, sigma):
#     d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
#     d2 = d1 - sigma*np.sqrt(T)
#     return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# def black_scholes_put(S0, K, T, r, sigma):
#     d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
#     d2 = d1 - sigma*np.sqrt(T)
#     return K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)

# # ------------------------------------------------------------------------------
# # 3) Monte Carlo Approaches
# # ------------------------------------------------------------------------------

# def mc_barrier_option_plain(S0, K, H, T, r, q, sigma, N, M, option_type):
#     """Plain MC (no variance reduction)."""
#     #np.random.seed(42)
#     dt   = T / M
#     disc = np.exp(-r*T)

#     is_call = 'call' in option_type.lower()
#     is_in   = 'in' in option_type.lower()
#     is_up   = 'up' in option_type.lower()

#     payoffs = np.zeros(N)
#     for i in range(N):
#         S_path = S0
#         barrier_breached = False
#         for _ in range(M):
#             z = np.random.randn()
#             S_path *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
#             if is_up and S_path >= H:
#                 barrier_breached = True
#             elif (not is_up) and S_path <= H:
#                 barrier_breached = True

#         if is_in:
#             if barrier_breached:
#                 payoff = max(S_path-K, 0) if is_call else max(K-S_path, 0)
#             else:
#                 payoff = 0.0
#         else:
#             if barrier_breached:
#                 payoff = 0.0
#             else:
#                 payoff = max(S_path-K, 0) if is_call else max(K-S_path, 0)
#         payoffs[i] = payoff

#     return disc*np.mean(payoffs)

# def mc_barrier_option_cv(S0, K, H, T, r, q, sigma, N, M, option_type):
#     """Control Variate (CV) only."""
#     #np.random.seed(42)
#     dt   = T / M
#     disc = np.exp(-r*T)
#     is_call = 'call' in option_type.lower()
#     is_in   = 'in' in option_type.lower()
#     is_up   = 'up' in option_type.lower()

#     # pick correct vanilla
#     if is_call:
#         vanilla_analytic = black_scholes_call(S0, K, T, r, sigma)
#     else:
#         vanilla_analytic = black_scholes_put(S0, K, T, r, sigma)

#     barrier_payoffs = np.zeros(N)
#     vanilla_payoffs = np.zeros(N)
#     for i in range(N):
#         S_path = S0
#         barrier_breached = False
#         for _ in range(M):
#             z = np.random.randn()
#             S_path *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
#             if is_up and S_path >= H:
#                 barrier_breached = True
#             elif (not is_up) and S_path <= H:
#                 barrier_breached = True

#         # Barrier payoff
#         if is_in:
#             payoff_barrier = max(S_path-K, 0) if (is_call and barrier_breached) else (max(K-S_path, 0) if (not is_call and barrier_breached) else 0.0)
#         else:
#             payoff_barrier = max(S_path-K, 0) if (is_call and not barrier_breached) else (max(K-S_path, 0) if (not is_call and not barrier_breached) else 0.0)
#         barrier_payoffs[i] = payoff_barrier

#         # Vanilla payoff
#         payoff_vanilla = max(S_path-K, 0) if is_call else max(K-S_path, 0)
#         vanilla_payoffs[i] = payoff_vanilla

#     mc_barrier = disc*np.mean(barrier_payoffs)
#     mc_vanilla = disc*np.mean(vanilla_payoffs)
#     mc_cv = mc_barrier + (vanilla_analytic - mc_vanilla)
#     return mc_cv

# def mc_barrier_option_av(S0, K, H, T, r, q, sigma, N, M, option_type):
#     """Antithetic Variates (AV) only."""
#     #np.random.seed(42)
#     dt   = T / M
#     disc = np.exp(-r*T)
#     is_call = 'call' in option_type.lower()
#     is_in   = 'in' in option_type.lower()
#     is_up   = 'up' in option_type.lower()

#     payoffs = np.zeros(N)
#     for i in range(N):
#         S_plus  = S0
#         S_minus = S0
#         breach_plus  = False
#         breach_minus = False

#         for _ in range(M):
#             z = np.random.randn()
#             # +z path
#             S_plus *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
#             if is_up and S_plus >= H:
#                 breach_plus = True
#             elif (not is_up) and S_plus <= H:
#                 breach_plus = True

#             # -z path
#             S_minus *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*(-z))
#             if is_up and S_minus >= H:
#                 breach_minus = True
#             elif (not is_up) and S_minus <= H:
#                 breach_minus = True

#         if is_in:
#             payoff_plus = max(S_plus-K, 0) if (is_call and breach_plus) else (max(K-S_plus, 0) if (not is_call and breach_plus) else 0.0)
#             payoff_minus= max(S_minus-K, 0) if (is_call and breach_minus) else (max(K-S_minus, 0) if (not is_call and breach_minus) else 0.0)
#         else:
#             payoff_plus = max(S_plus-K, 0) if (is_call and not breach_plus) else (max(K-S_plus, 0) if (not is_call and not breach_plus) else 0.0)
#             payoff_minus= max(S_minus-K, 0) if (is_call and not breach_minus) else (max(K-S_minus, 0) if (not is_call and not breach_minus) else 0.0)

#         payoffs[i] = 0.5*(payoff_plus + payoff_minus)

#     return disc*np.mean(payoffs)

# def mc_barrier_option_combined(S0, K, H, T, r, q, sigma, N, M, option_type):
#     """Combined: Antithetic Variates + Control Variate."""
#     #np.random.seed(42)
#     dt   = T / M
#     disc = np.exp(-r*T)
#     is_call = 'call' in option_type.lower()
#     is_up   = 'up' in option_type.lower()
#     is_in   = 'in' in option_type.lower()

#     # For the control variate
#     if is_call:
#         vanilla_analytic = black_scholes_call(S0, K, T, r, sigma)
#     else:
#         vanilla_analytic = black_scholes_put(S0, K, T, r, sigma)

#     barrier_payoffs = np.zeros(N)
#     vanilla_payoffs = np.zeros(N)
#     for i in range(N):
#         S_plus  = S0
#         S_minus = S0
#         breach_plus  = False
#         breach_minus = False
#         for _ in range(M):
#             z = np.random.randn()
#             S_plus *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
#             if is_up and S_plus >= H:
#                 breach_plus = True
#             elif (not is_up) and S_plus <= H:
#                 breach_plus = True

#             S_minus *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*(-z))
#             if is_up and S_minus >= H:
#                 breach_minus = True
#             elif (not is_up) and S_minus <= H:
#                 breach_minus = True

#         # barrier payoff
#         if is_in:
#             payoff_plus = max(S_plus-K,0) if (is_call and breach_plus) else (max(K-S_plus,0) if (not is_call and breach_plus) else 0.0)
#             payoff_minus= max(S_minus-K,0) if (is_call and breach_minus) else (max(K-S_minus,0) if (not is_call and breach_minus) else 0.0)
#         else:
#             payoff_plus = max(S_plus-K,0) if (is_call and not breach_plus) else (max(K-S_plus,0) if (not is_call and not breach_plus) else 0.0)
#             payoff_minus= max(S_minus-K,0) if (is_call and not breach_minus) else (max(K-S_minus,0) if (not is_call and not breach_minus) else 0.0)

#         avg_barrier = 0.5*(payoff_plus + payoff_minus)
#         barrier_payoffs[i] = avg_barrier

#         # For control variate, we only need the vanilla payoff from S_plus
#         if is_call:
#             payoff_vanilla = max(S_plus-K,0)
#         else:
#             payoff_vanilla = max(K-S_plus,0)
#         vanilla_payoffs[i] = payoff_vanilla

#     barrier_raw = disc*np.mean(barrier_payoffs)
#     vanilla_mc  = disc*np.mean(vanilla_payoffs)
#     barrier_combined = barrier_raw + (vanilla_analytic - vanilla_mc)
#     return barrier_combined

# # ------------------------------------------------------------------------------
# # 4) Streamlit UI
# # ------------------------------------------------------------------------------

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

# st.sidebar.write("MC Steps (M) and # of Paths (N list)")
# M = st.sidebar.number_input("Time Steps (M)", min_value=1, value=100)
# default_N = "100,1000,10000"
# N_str = st.sidebar.text_input("Comma-separated N values:", default_N)
# N_list = []
# try:
#     N_list = sorted([int(x.strip()) for x in N_str.split(",") if int(x.strip())>0])
# except:
#     pass

# tab1, tab2 = st.tabs(["Closed-Form", "MC Analysis"])

# with tab1:
#     st.title("Closed-Form (Analytical) Price")
#     cf_price = barrier_option_price(S0, K, T, r, q, sigma, H, option_type)
#     if cf_price is not None:
#         st.subheader(f"{option_type.capitalize()} Price = {cf_price:.4f}")
#     else:
#         st.write("No closed-form formula implemented or invalid parameters.")

# with tab2:
#     st.title("Monte Carlo Comparison (All on One Graph)")

#     if st.button("Run MC"):
#         # Build a DataFrame with columns: N, Plain, CV, AV, Combined
#         results_plain = []
#         results_cv    = []
#         results_av    = []
#         results_comb  = []
#         plain_error = []
#         cv_error = []
#         av_error = []
#         comb_error = []

#         for N in N_list:
#             # 1) Plain
#             mc_plain = mc_barrier_option_plain(S0, K, H, T, r, q, sigma, N, M, option_type)
#             # 2) CV
#             mc_cv = mc_barrier_option_cv(S0, K, H, T, r, q, sigma, N, M, option_type)
#             # 3) AV
#             mc_av = mc_barrier_option_av(S0, K, H, T, r, q, sigma, N, M, option_type)
#             # 4) Combined
#             mc_comb = mc_barrier_option_combined(S0, K, H, T, r, q, sigma, N, M, option_type)
#             # 5) Analytical
#             analytical_price = barrier_option_price(S0, K, T, r, q, sigma, H, option_type)
            
#             results_plain.append(mc_plain)
#             results_cv.append(mc_cv)
#             results_av.append(mc_av)
#             results_comb.append(mc_comb)
#             plain_error.append(np.abs(mc_plain-analytical_price))
#             cv_error.append(np.abs(mc_cv-analytical_price))
#             av_error.append(np.abs(mc_av-analytical_price))
#             comb_error.append(np.abs(mc_comb-analytical_price))

#         df = pd.DataFrame({
#             "N": N_list,
#             "Plain": results_plain,
#             "CV": results_cv,
#             "AV": results_av,
#             "Combined": results_comb,
#             "Plain MC Error": plain_error,
#             "CV Error": cv_error,
#             "AV Error": av_error,
#             "Combined Error": comb_error 
#         })

#         # Single plot: We'll show Plain, CV, AV, Combined all together
#         fig = px.line(
#             df, 
#             x="N", 
#             y=["Plain", "CV", "AV", "Combined"], 
#             markers=True,
#             title=f"{option_type.capitalize()} - MC Methods Comparison"
#         )
        
#         fig.update_layout(
#             title={
#             "text": f"{option_type.capitalize()} - MC Methods Comparison",
#             "x": 0.5,
#             "xanchor": "center"
#             }
# )
        
#         fig.add_hline(
#             y=analytical_price, 
#             line_dash="dash", 
#             line_color="red",
#             annotation_text=(f"Analytical={analytical_price:.4f}")
#         )

#         # Add horizontal line for closed-form if available
#         if cf_price is not None:
#             fig.add_hline(
#                 y=cf_price, 
#                 line_dash="dash", 
#                 line_color="red",
#                 annotation_text=f"Analytical={cf_price:.4f}"
#             )

#         st.plotly_chart(fig, use_container_width=True)

#         st.write("### Full Results Table")
#         st.dataframe(df.style.format("{:.4f}"))

#         if cf_price is not None:
#             st.write(f"**Closed-Form** = {cf_price:.4f}")

#     else:
#         st.info("Click 'Run MC' to compare all methods on a single plot.")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import norm

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

# --------------------------------------------------------------------------
# 4) Streamlit UI
# --------------------------------------------------------------------------

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

st.sidebar.write("MC Steps (M) and # of Paths (N list)")
M = st.sidebar.number_input("Time Steps (M)", min_value=1, value=100)
default_N = "100,1000,10000"
N_str = st.sidebar.text_input("Comma-separated N values:", default_N)
N_list = []
try:
    N_list = sorted([int(x.strip()) for x in N_str.split(",") if int(x.strip())>0])
except:
    pass

tab1, tab2 = st.tabs(["Closed-Form", "MC Analysis"])

with tab1:
    st.title("Closed-Form (Analytical) Price")
    cf_price = barrier_option_price(S0, K, T, r, q, sigma, H, option_type)
    if cf_price is not None:
        st.subheader(f"{option_type.capitalize()} Price = {cf_price:.4f}")
    else:
        st.write("No closed-form formula implemented or invalid parameters.")

with tab2:
    st.title("Monte Carlo Comparison")

    if st.button("Run MC"):
        # Build a DataFrame with columns: N, Plain, CV, AV, Combined
        results_plain = []
        results_cv    = []
        results_av    = []
        results_comb  = []
        plain_error = []
        cv_error    = []
        av_error    = []
        comb_error  = []

        analytical_price = barrier_option_price(S0, K, T, r, q, sigma, H, option_type)

        for N in N_list:
            # 1) Plain
            mc_plain = mc_barrier_option_plain(S0, K, H, T, r, q, sigma, N, M, option_type)
            # 2) CV
            mc_cv = mc_barrier_option_cv(S0, K, H, T, r, q, sigma, N, M, option_type)
            # 3) AV
            mc_av = mc_barrier_option_av(S0, K, H, T, r, q, sigma, N, M, option_type)
            # 4) Combined
            mc_comb = mc_barrier_option_combined(S0, K, H, T, r, q, sigma, N, M, option_type)

            # Store results
            results_plain.append(mc_plain)
            results_cv.append(mc_cv)
            results_av.append(mc_av)
            results_comb.append(mc_comb)

            # Errors (absolute difference vs. analytical)
            plain_error.append(np.abs(mc_plain - analytical_price))
            cv_error.append(np.abs(mc_cv - analytical_price))
            av_error.append(np.abs(mc_av - analytical_price))
            comb_error.append(np.abs(mc_comb - analytical_price))

        df = pd.DataFrame({
            "N": N_list,
            "Plain": results_plain,
            "CV": results_cv,
            "AV": results_av,
            "Combined": results_comb,
            "Plain MC Error": plain_error,
            "CV Error": cv_error,
            "AV Error": av_error,
            "Combined Error": comb_error 
        })

        # -- FIGURE 1: Price Estimates --
        fig_estimates = px.line(
            df.melt(id_vars="N", value_vars=["Plain","CV","AV","Combined"],
                    var_name="Method", value_name="Price Estimate"),
            x="N", y="Price Estimate", color="Method", markers=True,
            title=f"{option_type.capitalize()} - MC Methods: Price Estimates",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_estimates.update_layout(
            title={"x":0.5, "xanchor":"center"},
            legend_title_text="MC Method"
        )
        # Add horizontal line for closed-form (if available)
        if analytical_price is not None:
            fig_estimates.add_hline(
                y=analytical_price,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Analytical={analytical_price:.4f}",
                annotation_position="top right"
            )

        st.plotly_chart(fig_estimates, use_container_width=True)

        # -- FIGURE 2: Errors --
        fig_errors = px.line(
            df.melt(id_vars="N", value_vars=["Plain MC Error","CV Error","AV Error","Combined Error"],
                    var_name="Method", value_name="Error"),
            x="N", y="Error", color="Method", markers=True,
            title=f"{option_type.capitalize()} - MC Methods: Absolute Errors",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_errors.update_layout(
            title={"x":0.5, "xanchor":"center"},
            legend_title_text="MC Method"
        )
        st.plotly_chart(fig_errors, use_container_width=True)

        # -- Table Display with Highlights --
        st.write("### Full Results Table")

        # Helper to highlight the minimum error(s) in each row
        def highlight_min_error(row):
            error_cols = ["Plain MC Error", "CV Error", "AV Error", "Combined Error"]
            # True/False for which columns match the min error in this row
            is_min = row[error_cols] == row[error_cols].min()
            return ["background-color: #ffff66" if v else "" for v in is_min]

        df_style = df.style.format("{:.4f}") \
                           .apply(highlight_min_error, axis=1, subset=["Plain MC Error", "CV Error", "AV Error", "Combined Error"])
        
        st.dataframe(df_style, use_container_width=True)

        if analytical_price is not None:
            st.markdown(f"**Closed-Form Analytical Price:** `{analytical_price:.4f}`")
    else:
        st.info("Click 'Run MC' to compare all methods on a single plot.")

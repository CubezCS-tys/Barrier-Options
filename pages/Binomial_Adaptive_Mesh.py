
# import streamlit as st
# import numpy as np
# import math
# import time
# from scipy.stats import norm

# def calc_d1(S0, K, r, q, sigma, T):
#     return (np.log(S0 / K) + (r - q + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))

# def calc_d2(S0, K, r, q, sigma, T):
#     return calc_d1(S0, K, r, q, sigma, T) - sigma * np.sqrt(T)

# def calc_c(S0, K, r, q, sigma, T):
#     """Analytical price of a plain vanilla Call via Black-Scholes."""
#     d1 = calc_d1(S0, K, r, q, sigma, T)
#     d2 = calc_d2(S0, K, r, q, sigma, T)
#     return (S0 * np.exp(-q*T)*norm.cdf(d1)
#             - K * np.exp(-r*T)*norm.cdf(d2))

# def calc_p(S0, K, r, q, sigma, T):
#     """Analytical price of a plain vanilla Put via Black-Scholes."""
#     d1 = calc_d1(S0, K, r, q, sigma, T)
#     d2 = calc_d2(S0, K, r, q, sigma, T)
#     return (K * np.exp(-r*T)*norm.cdf(-d2)
#             - S0 * np.exp(-q*T)*norm.cdf(-d1))

# def calc_lambda(r, q, sigma):
#     """λ = (r - q + σ²/2) / σ²"""
#     return (r - q + 0.5 * sigma**2) / (sigma**2)

# def calc_y(barrier, S0, K, T, sigma, r, q):
#     """
#     y = ln(barrier^2/(S0*K)) / (sigma*sqrt(T)) + λ*sigma*sqrt(T)
#     """
#     lam = calc_lambda(r, q, sigma)
#     return (np.log((barrier**2)/(S0*K)) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

# def calc_x1(S0, barrier, T, sigma, r, q):
#     """
#     x1 = ln(S0/barrier)/(sigma*sqrt(T)) + λ*sigma*sqrt(T)
#     """
#     lam = calc_lambda(r, q, sigma)
#     return (np.log(S0/barrier) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

# def calc_y1(S0, barrier, T, sigma, r, q):
#     """
#     y1 = ln(barrier/S0)/(sigma*sqrt(T)) + λ*sigma*sqrt(T)
#     """
#     lam = calc_lambda(r, q, sigma)
#     return (np.log(barrier/S0) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

# def black_scholes(S, K, T, r, sigma, option_type):
#     """Plain vanilla Black-Scholes for a Call or Put."""
#     d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
#     if option_type.lower() == "call":
#         return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
#     else:
#         return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# def barrier_option_price(S0, K, T, r, q, sigma, barrier, option_type):
#     """
#     Analytical price of various knock-in/out barrier options.
#     option_type examples: 'down-and-in call', 'down-and-out call',
#                           'up-and-in call',   'up-and-out call', etc.
#     """
#     x1 = calc_x1(S0, barrier, T, sigma, r, q)
#     y1 = calc_y1(S0, barrier, T, sigma, r, q)
#     c  = calc_c(S0, K, r, q, sigma, T)
#     p  = calc_p(S0, K, r, q, sigma, T)
#     lam = calc_lambda(r, q, sigma)
#     y   = calc_y(barrier, S0, K, T, sigma, r, q)

#     # (The code below is exactly as you provided; handles many cases.)
#     # ----------------------------------------------------------------
#     # Down-and-in Call
#     # ----------------------------------------------------------------
#     if option_type == 'down-and-in call' and barrier <= K and S0 <= barrier:
#         vanilla = black_scholes(S0, K, T, r, sigma, "call")
#         return vanilla
#     elif option_type == 'down-and-in call' and barrier <= K:
#         cdi = (S0 * np.exp(-q*T) * (barrier/S0)**(2*lam) * norm.cdf(y)
#                - K * np.exp(-r*T) * (barrier/S0)**(2*lam - 2)
#                  * norm.cdf(y - sigma*np.sqrt(T)))
#         return cdi
#     elif option_type == 'down-and-in call' and barrier >= K:
#         term1 = S0*np.exp(-q*T)*norm.cdf(x1)
#         term2 = K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
#         term3 = S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(y1)
#         term4 = K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)*norm.cdf(y1 - sigma*np.sqrt(T))
#         cdo   = term1 - term2 - term3 + term4
#         if cdo < 0:
#             cdo = 0
#         cdi = c - cdo
#         return cdi

#     # ----------------------------------------------------------------
#     # Down-and-out Call
#     # ----------------------------------------------------------------
#     elif option_type == 'down-and-out call' and barrier <= K:
#         cdi = (S0 * np.exp(-q*T) * (barrier/S0)**(2*lam) * norm.cdf(y)
#                - K * np.exp(-r*T) * (barrier/S0)**(2*lam - 2)
#                  * norm.cdf(y - sigma*np.sqrt(T)))
#         cdo = c - cdi
#         return max(cdo, 0)
#     elif option_type == 'down-and-out call' and barrier >= K:
#         term1 = S0 * np.exp(-q*T)*norm.cdf(x1)
#         term2 = K  * np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
#         term3 = S0 * np.exp(-q*T)*((barrier/S0)**(2*lam))*norm.cdf(y1)
#         term4 = K  * np.exp(-r*T)*((barrier/S0)**(2*lam - 2))*norm.cdf(y1 - sigma*np.sqrt(T))
#         cdo   = term1 - term2 - term3 + term4
#         return max(cdo, 0)

#     # ----------------------------------------------------------------
#     # Up-and-in Call
#     # ----------------------------------------------------------------
#     elif option_type == 'up-and-in call' and barrier > K:
#         cui = (S0*np.exp(-q*T)*norm.cdf(x1)
#                - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
#                - S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*(norm.cdf(-y) - norm.cdf(-y1))
#                + K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)
#                  * (norm.cdf(-y + sigma*np.sqrt(T))
#                     - norm.cdf(-y1 + sigma*np.sqrt(T))))
#         return cui
#     elif option_type == 'up-and-in call' and barrier <= K:
#         return c

#     # ----------------------------------------------------------------
#     # Up-and-out Call
#     # ----------------------------------------------------------------
#     elif option_type == 'up-and-out call' and barrier <= K:
#         # Usually worthless if barrier <= K and S0 is above barrier,
#         # but let's keep your logic consistent:
#         return 0.0
#     elif option_type == 'up-and-out call' and barrier > K:
#         cui = (S0*np.exp(-q*T)*norm.cdf(x1)
#                - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
#                - S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*(norm.cdf(-y) - norm.cdf(-y1))
#                + K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)
#                  * (norm.cdf(-y + sigma*np.sqrt(T))
#                     - norm.cdf(-y1 + sigma*np.sqrt(T))))
#         cuo = c - cui
#         return max(cuo, 0)

#     # ----------------------------------------------------------------
#     # Up-and-in Put
#     # ----------------------------------------------------------------
#     elif option_type == 'up-and-in put' and barrier >= K and barrier <= S0:
#         pui = black_scholes(S0, K, T, r, sigma, "put")
#         return pui
#     elif option_type == 'up-and-in put' and barrier >= K:
#         pui = (-S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(-y)
#                + K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)
#                  * norm.cdf(-y + sigma*np.sqrt(T)))
#         return pui
#     elif option_type == 'up-and-in put' and barrier <= K:
#         # up-and-in put = p - up-and-out put in some references,
#         # but let's keep your code:
#         return p

#     # ----------------------------------------------------------------
#     # Up-and-out Put
#     # ----------------------------------------------------------------
#     elif option_type == 'up-and-out put' and barrier >= K:
#         pui = (-S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(-y)
#                + K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)
#                  * norm.cdf(-y + sigma*np.sqrt(T)))
#         puo = p - pui
#         return max(puo, 0)
#     elif option_type == 'up-and-out put' and barrier <= K:
#         puo = (
#             -S0*np.exp(-q*T)*norm.cdf(-x1)
#             + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
#             + S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(-y1)
#             - K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)*norm.cdf(-y1 + sigma*np.sqrt(T))
#         )
#         return max(puo, 0)

#     # ----------------------------------------------------------------
#     # Down-and-in Put
#     # ----------------------------------------------------------------
#     elif option_type == 'down-and-in put' and barrier < K and S0 < barrier:
#         vanilla = black_scholes(S0, K, T, r, sigma, "put")
#         return vanilla
#     elif option_type == 'down-and-in put' and barrier > K:
#         return p
#     elif option_type == 'down-and-in put' and barrier < K:
#         pdi = (
#             -S0*np.exp(-q*T)*norm.cdf(-x1)
#             + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
#             + S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*(norm.cdf(y) - norm.cdf(y1))
#             - K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)
#               * (norm.cdf(y - sigma*np.sqrt(T))
#                  - norm.cdf(y1 - sigma*np.sqrt(T)))
#         )
#         return pdi

#     # ----------------------------------------------------------------
#     # Down-and-out Put
#     # ----------------------------------------------------------------
#     elif option_type == 'down-and-out put' and barrier > K:
#         return 0
#     elif option_type == 'down-and-out put' and barrier < K:
#         pdi = (
#             -S0*np.exp(-q*T)*norm.cdf(-x1)
#             + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
#             + S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*(norm.cdf(y) - norm.cdf(y1))
#             - K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)
#               * (norm.cdf(y - sigma*np.sqrt(T))
#                  - norm.cdf(y1 - sigma*np.sqrt(T)))
#         )
#         pdo = p - pdi
#         return max(pdo, 0)

#     # Fallback
#     return None


# def barrier_binomial_option_price(S0, K, r, q_div, T, sigma, steps,
#                                   barrier_option_type, H,
#                                   option_side='call', rebate=0.0):
#     """
#     Binomial pricing of a barrier option, tracking whether the barrier is hit.
#     Incorporates continuous dividend yield q_div similarly.
#     """
#     dt = T / steps
#     u = math.exp(sigma * math.sqrt(dt))
#     d = 1.0 / u
#     disc = math.exp(-r * dt)
#     m = math.exp((r - q_div) * dt)
#     p_up = (m - d) / (u - d)

#     # Determine barrier direction
#     if barrier_option_type.lower().startswith("up"):
#         barrier_direction = "up"
#     else:
#         barrier_direction = "down"
    
#     def intrinsic(S):
#         if option_side.lower() == 'call':
#             return max(S - K, 0)
#         else:
#             return max(K - S, 0)

#     # Barrier type
#     is_knock_out = barrier_option_type.lower().endswith("out")
#     is_knock_in  = barrier_option_type.lower().endswith("in")

#     memo = {}
#     # f(i, j, bh): value at node i, j up-moves so far, bh=barrier hit?
#     def f(i, j, bh):
#         key = (i, j, bh)
#         if key in memo:
#             return memo[key]
        
#         # Current price
#         S = S0 * (u**j) * (d**(i-j))
#         t = i * dt
        
#         # If at final step
#         if i == steps:
#             if is_knock_out:
#                 # If barrier was hit, payoff=rebate; else=intrinsic
#                 val = rebate if bh else intrinsic(S)
#             else:
#                 # Knock-in: payoff only if barrier was hit
#                 val = intrinsic(S) if bh else 0.0
#             memo[key] = val
#             return val
        
#         # If knocked out already, payoff=PV of rebate
#         if is_knock_out and bh:
#             val = rebate * math.exp(-r * (T - t))
#             memo[key] = val
#             return val

#         # Next step up/down
#         S_up = S0*(u**(j+1))*(d**((i+1)-(j+1)))
#         S_down = S0*(u**j)*(d**((i+1)-j))

#         def barrier_hit(S_new, current_bh):
#             if current_bh:
#                 return True
#             if barrier_direction == "up":
#                 return S_new >= H
#             else:
#                 return S_new <= H

#         new_bh_up   = barrier_hit(S_up, bh)
#         new_bh_down = barrier_hit(S_down, bh)

#         # If knock-out and barrier is hit on the next move:
#         if is_knock_out and new_bh_up:
#             val_up = rebate * math.exp(-r*(T-(t+dt)))
#         else:
#             val_up = f(i+1, j+1, new_bh_up)
        
#         if is_knock_out and new_bh_down:
#             val_down = rebate * math.exp(-r*(T-(t+dt)))
#         else:
#             val_down = f(i+1, j, new_bh_down)

#         val = disc * (p_up * val_up + (1 - p_up) * val_down)
#         memo[key] = val
#         return val

#     return f(0, 0, False)

# # Adaptive Mesh Refinement (AMR) Binomial
# # def adaptive_barrier_binomial(S0, K, r, q, T, sigma, coarse_steps, fine_steps, barrier, barrier_option_type, option_side, rebate=0.0, fine_region=0.1):
# #     critical_region = (barrier*(1-fine_region), barrier*(1+fine_region))
# #     dt_fine = T / fine_steps
# #     dt_coarse = T / coarse_steps

# #     u_fine = np.exp(sigma * np.sqrt(dt_fine))
# #     d_fine = 1/u_fine
# #     disc_fine = np.exp(-r * dt_fine)
# #     p_fine = (np.exp((r - q)*dt_fine) - d_fine) / (u_fine - d_fine)

# #     u_coarse = np.exp(sigma * np.sqrt(dt_coarse))
# #     d_coarse = 1/u_coarse
# #     disc_coarse = np.exp(-r * dt_coarse)
# #     p_coarse = (np.exp((r - q)*dt_coarse) - d_coarse) / (u_coarse - d_coarse)

# #     memo = {}
# #     def intrinsic(S):
# #         return max(S - K, 0) if option_side == 'call' else max(K - S, 0)

# #     def adaptive_f(S, t, bh):
# #         if t >= T:
# #             return rebate if ("out" in barrier_option_type and bh) else (intrinsic(S) if ("in" in barrier_option_type and bh) or ("out" in barrier_option_type and not bh) else 0)

# #         is_fine = critical_region[0] <= S <= critical_region[1]
# #         dt, u, d, disc, p = (dt_fine, u_fine, d_fine, disc_fine, p_fine) if is_fine else (dt_coarse, u_coarse, d_coarse, disc_coarse, p_coarse)

# #         barrier_hit = bh or (S >= barrier if "up" in barrier_option_type else S <= barrier)
# #         S_up, S_down = S*u, S*d

# #         key = (round(S,4), round(t,4), barrier_hit)
# #         if key in memo:
# #             return memo[key]

# #         val_up = adaptive_f(S_up, t+dt, barrier_hit)
# #         val_down = adaptive_f(S_down, t+dt, barrier_hit)
# #         val = disc * (p*val_up + (1-p)*val_down)

# #         memo[key] = val
# #         return val

# #     return adaptive_f(S0, 0, False)

# def adaptive_barrier_binomial(
#     S0, K, r, q, T, sigma,
#     coarse_steps, fine_steps,
#     barrier, barrier_option_type, option_side,
#     rebate=0.0, fine_region=0.1
# ):
#     # Define critical region around the barrier for using fine steps
#     critical_region = (barrier * (1 - fine_region), barrier * (1 + fine_region))

#     # Default time steps
#     dt_fine = T / fine_steps
#     dt_coarse = T / coarse_steps

#     # Binomial params for fine grid
#     u_fine = np.exp(sigma * np.sqrt(dt_fine))
#     d_fine = 1 / u_fine
#     disc_fine = np.exp(-r * dt_fine)
#     p_fine = (np.exp((r - q) * dt_fine) - d_fine) / (u_fine - d_fine)

#     # Binomial params for coarse grid
#     u_coarse = np.exp(sigma * np.sqrt(dt_coarse))
#     d_coarse = 1 / u_coarse
#     disc_coarse = np.exp(-r * dt_coarse)
#     p_coarse = (np.exp((r - q) * dt_coarse) - d_coarse) / (u_coarse - d_coarse)

#     memo = {}

#     def intrinsic(S):
#         return max(S - K, 0) if option_side == 'call' else max(K - S, 0)

#     def adaptive_f(S, t, bh):
#         # Base case: maturity reached
#         if t >= T:
#             if "out" in barrier_option_type:
#                 return rebate if bh else intrinsic(S)
#             elif "in" in barrier_option_type:
#                 return intrinsic(S) if bh else 0
#             else:
#                 return intrinsic(S)

#         # Decide which time step to use (fine or coarse)
#         is_fine = critical_region[0] <= S <= critical_region[1]
#         dt = dt_fine if is_fine else dt_coarse

#         # --- SNAP LOGIC: adjust dt if it overshoots maturity ---
#         if t + dt > T:
#             dt = T - t
#             u = np.exp(sigma * np.sqrt(dt))
#             d = 1 / u
#             disc = np.exp(-r * dt)
#             p = (np.exp((r - q) * dt) - d) / (u - d)
#         else:
#             u, d = (u_fine, d_fine) if is_fine else (u_coarse, d_coarse)
#             disc = disc_fine if is_fine else disc_coarse
#             p = p_fine if is_fine else p_coarse

#         # Update barrier hit status
#         barrier_hit = bh or (S >= barrier if "up" in barrier_option_type else S <= barrier)

#         S_up, S_down = S * u, S * d
#         key = (round(S, 4), round(t, 6), barrier_hit)
#         if key in memo:
#             return memo[key]

#         val_up = adaptive_f(S_up, t + dt, barrier_hit)
#         val_down = adaptive_f(S_down, t + dt, barrier_hit)
#         val = disc * (p * val_up + (1 - p) * val_down)

#         memo[key] = val
#         return val

#     return adaptive_f(S0, 0, False)


# # Helper to unify barrier type + side for the analytical function
# def combine_barrier_and_side(barrier_option_type, side):
#     # e.g. "Up-and-Out" + "call" -> "up-and-out call"
#     return f"{barrier_option_type.lower().replace(' ', '-')} {side.lower()}"


# # Streamlit UI
# st.title("Adaptive Mesh Refinement for Barrier Options (Binomial Model)")
# col1, col2 = st.columns(2)
# with col1:
#     st.header("Model Parameters")
#     S0 = st.number_input("Initial Price S0", value = 100.0, min_value=0.0)
#     K = st.number_input("Strike Price K", value = 100.0, min_value=0.0)
#     r = st.number_input("Risk-free Rate r", value = 0.05, min_value=0.0)
#     q = st.number_input("Dividend Yield q", value = 0.0, min_value=0.0)
#     T = st.number_input("Time to Maturity T", value = 1.0, min_value=0.0)
#     sigma = st.number_input("Volatility σ", value = 0.2, min_value=0.0)
#     barrier = st.number_input("Barrier Level", value = 110.0, min_value=0.0)
#     option_side = st.selectbox("Option Side", ["call", "put"])
#     barrier_type = st.selectbox("Barrier Type", ["Up-and-Out", "Down-and-Out", "Up-and-In", "Down-and-In"])

# with col2:
#     st.header("AMR Parameters")
#     coarse_steps = st.number_input("Coarse Steps", value = 20, min_value = 0)
#     fine_steps = st.number_input("Fine Steps (around barrier)", value = 400, min_value = 0)
#     fine_region = st.slider("Fine Region Width (%) around Barrier", 0.01, 0.3, 0.1)

# if st.button("Calculate"):
#     start = time.time()
#     adaptive_price = adaptive_barrier_binomial(S0, K, r, q, T, sigma, coarse_steps, fine_steps, barrier, barrier_type.lower(), option_side, 0.0, fine_region)
#     adaptive_time = time.time() - start

#     start = time.time()
#     regular_price = barrier_binomial_option_price(S0, K, r, q, T, sigma, coarse_steps, barrier_type.lower(), barrier, option_side, 0.0)
#     regular_time = time.time() - start

#     cf_type = combine_barrier_and_side(barrier_type, option_side)
#     an_price = barrier_option_price(S0, K, T, r, q, sigma, barrier, cf_type)



#     st.subheader("Results Comparison")
#     st.write(f"Adaptive Binomial Price: **{adaptive_price:.4f}** (Time: {adaptive_time:.4f}s)")
#     st.write(f"Regular Binomial Price: **{regular_price:.4f}** (Time: {regular_time:.4f}s)")
#     #st.write(f"Difference (Adaptive - Regular): **{adaptive_price - regular_price:.4e}**")
#     st.write(f"Analytical Price: **{an_price:.4f}**")

#     st.subheader("Interpretation")
#     st.markdown("""
#     - **Adaptive method** should yield more accurate results near barriers.
#     - Compare computational times and errors.
#     - Test various mesh refinements to balance efficiency and accuracy.
#     """)



# import plotly.graph_objects as go

# # Error vs Stock Price (S0) plot
# st.subheader("Error vs. Stock Price (S0)")

# # Inputs for error analysis
# S0_min = st.number_input("Minimum S0", value=50.0, step=1.0, key='min_s0')
# S0_max = st.number_input("Maximum S0", value=150.0, step=1.0, key='max_s0')
# S0_step = st.number_input("Increment for S0", value=5.0, step=1.0, key='step_s0')

# if st.button("Generate Error Plot"):
#     S0_values = np.arange(S0_min, S0_max + S0_step, S0_step)

#     adaptive_errors = []  # errors for adaptive mesh
#     regular_errors = []   # errors for regular binomial

#     for s in S0_values:
#         adaptive_price = adaptive_barrier_binomial(s, K, r, q, T, sigma,
#                                                    coarse_steps, fine_steps,
#                                                    barrier, barrier_type.lower(),
#                                                    option_side, 0.0, fine_region)

#         regular_price = barrier_binomial_option_price(s, K, r, q, T, sigma,
#                                                       coarse_steps, barrier_type.lower(),
#                                                       barrier, option_side, 0.0)

#         cf_type = combine_barrier_and_side(barrier_type, option_side)
#         an_price = barrier_option_price(s, K, T, r, q, sigma, barrier, cf_type)

#         if an_price is not None:
#             adaptive_err = np.abs(adaptive_price - an_price)
#             regular_err = np.abs(regular_price - an_price)
#         else:
#             adaptive_err = np.nan
#             regular_err = np.nan

#         adaptive_errors.append(adaptive_err)
#         regular_errors.append(regular_err)

#     # Plotting
#     fig_err = go.Figure()
#     fig_err.add_trace(go.Scatter(
#         x=S0_values, y=adaptive_errors,
#         mode='lines+markers',
#         line=dict(shape='linear', width=2),
#         marker=dict(size=6),
#         name='Adaptive Binomial'
#     ))

#     fig_err.add_trace(go.Scatter(
#         x=S0_values, y=regular_errors,
#         mode='lines+markers',
#         line=dict(shape='linear', width=2),
#         marker=dict(size=6),
#         name='Regular Binomial'
#     ))

#     fig_err.update_layout(
#         title="Error vs. Stock Price (S0)",
#         xaxis_title="Stock Price (S0)",
#         yaxis_title="Error",
#         template="simple_white"
#     )

#     st.plotly_chart(fig_err, use_container_width=True)

import streamlit as st
import numpy as np
import math
import time
import plotly.graph_objects as go
from scipy.stats import norm

# -------------------- Option Pricing Functions --------------------

def calc_d1(S0, K, r, q, sigma, T):
    return (np.log(S0 / K) + (r - q + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))

def calc_d2(S0, K, r, q, sigma, T):
    return calc_d1(S0, K, r, q, sigma, T) - sigma * np.sqrt(T)

def calc_c(S0, K, r, q, sigma, T):
    """Analytical price of a plain vanilla Call via Black-Scholes."""
    d1 = calc_d1(S0, K, r, q, sigma, T)
    d2 = calc_d2(S0, K, r, q, sigma, T)
    return (S0 * np.exp(-q*T)*norm.cdf(d1)
            - K * np.exp(-r*T)*norm.cdf(d2))

def calc_p(S0, K, r, q, sigma, T):
    """Analytical price of a plain vanilla Put via Black-Scholes."""
    d1 = calc_d1(S0, K, r, q, sigma, T)
    d2 = calc_d2(S0, K, r, q, sigma, T)
    return (K * np.exp(-r*T)*norm.cdf(-d2)
            - S0 * np.exp(-q*T)*norm.cdf(-d1))

def calc_lambda(r, q, sigma):
    """λ = (r - q + σ²/2) / σ²"""
    return (r - q + 0.5 * sigma**2) / (sigma**2)

def calc_y(barrier, S0, K, T, sigma, r, q):
    """
    y = ln(barrier^2/(S0*K)) / (sigma*sqrt(T)) + λ*sigma*sqrt(T)
    """
    lam = calc_lambda(r, q, sigma)
    return (np.log((barrier**2)/(S0*K)) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

def calc_x1(S0, barrier, T, sigma, r, q):
    """
    x1 = ln(S0/barrier)/(sigma*sqrt(T)) + λ*sigma*sqrt(T)
    """
    lam = calc_lambda(r, q, sigma)
    return (np.log(S0/barrier) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

def calc_y1(S0, barrier, T, sigma, r, q):
    """
    y1 = ln(barrier/S0)/(sigma*sqrt(T)) + λ*sigma*sqrt(T)
    """
    lam = calc_lambda(r, q, sigma)
    return (np.log(barrier/S0) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

def black_scholes(S, K, T, r, sigma, option_type):
    """Plain vanilla Black-Scholes for a Call or Put."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def barrier_option_price(S0, K, T, r, q, sigma, barrier, option_type):
    """
    Analytical price of various knock-in/out barrier options.
    option_type examples: 'down-and-in call', 'down-and-out call',
                          'up-and-in call',   'up-and-out call', etc.
    """
    x1 = calc_x1(S0, barrier, T, sigma, r, q)
    y1 = calc_y1(S0, barrier, T, sigma, r, q)
    c  = calc_c(S0, K, r, q, sigma, T)
    p  = calc_p(S0, K, r, q, sigma, T)
    lam = calc_lambda(r, q, sigma)
    y   = calc_y(barrier, S0, K, T, sigma, r, q)

    # Down-and-in Call
    if option_type == 'down-and-in call' and barrier <= K and S0 <= barrier:
        vanilla = black_scholes(S0, K, T, r, sigma, "call")
        return vanilla
    elif option_type == 'down-and-in call' and barrier <= K:
        cdi = (S0 * np.exp(-q*T) * (barrier/S0)**(2*lam) * norm.cdf(y)
               - K * np.exp(-r*T) * (barrier/S0)**(2*lam - 2)
                 * norm.cdf(y - sigma*np.sqrt(T)))
        return cdi
    elif option_type == 'down-and-in call' and barrier >= K:
        term1 = S0*np.exp(-q*T)*norm.cdf(x1)
        term2 = K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
        term3 = S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(y1)
        term4 = K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)*norm.cdf(y1 - sigma*np.sqrt(T))
        cdo   = term1 - term2 - term3 + term4
        if cdo < 0:
            cdo = 0
        cdi = c - cdo
        return cdi

    # Down-and-out Call
    elif option_type == 'down-and-out call' and barrier <= K:
        cdi = (S0 * np.exp(-q*T) * (barrier/S0)**(2*lam) * norm.cdf(y)
               - K * np.exp(-r*T) * (barrier/S0)**(2*lam - 2)
                 * norm.cdf(y - sigma*np.sqrt(T)))
        cdo = c - cdi
        return max(cdo, 0)
    elif option_type == 'down-and-out call' and barrier >= K:
        term1 = S0 * np.exp(-q*T)*norm.cdf(x1)
        term2 = K  * np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
        term3 = S0 * np.exp(-q*T)*((barrier/S0)**(2*lam))*norm.cdf(y1)
        term4 = K  * np.exp(-r*T)*((barrier/S0)**(2*lam - 2))*norm.cdf(y1 - sigma*np.sqrt(T))
        cdo   = term1 - term2 - term3 + term4
        return max(cdo, 0)

    # Up-and-in Call
    elif option_type == 'up-and-in call' and barrier > K:
        cui = (S0*np.exp(-q*T)*norm.cdf(x1)
               - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
               - S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*(norm.cdf(-y) - norm.cdf(-y1))
               + K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)
                 * (norm.cdf(-y + sigma*np.sqrt(T))
                    - norm.cdf(-y1 + sigma*np.sqrt(T))))
        return cui
    elif option_type == 'up-and-in call' and barrier <= K:
        return c

    # Up-and-out Call
    elif option_type == 'up-and-out call' and barrier <= K:
        return 0.0
    elif option_type == 'up-and-out call' and barrier > K:
        cui = (S0*np.exp(-q*T)*norm.cdf(x1)
               - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
               - S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*(norm.cdf(-y) - norm.cdf(-y1))
               + K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)
                 * (norm.cdf(-y + sigma*np.sqrt(T))
                    - norm.cdf(-y1 + sigma*np.sqrt(T))))
        cuo = c - cui
        return max(cuo, 0)

    # Up-and-in Put
    elif option_type == 'up-and-in put' and barrier >= K and barrier <= S0:
        pui = black_scholes(S0, K, T, r, sigma, "put")
        return pui
    elif option_type == 'up-and-in put' and barrier >= K:
        pui = (-S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(-y)
               + K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)
                 * norm.cdf(-y + sigma*np.sqrt(T)))
        return pui
    elif option_type == 'up-and-in put' and barrier <= K:
        return p

    # Up-and-out Put
    elif option_type == 'up-and-out put' and barrier >= K:
        pui = (-S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(-y)
               + K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)
                 * norm.cdf(-y + sigma*np.sqrt(T)))
        puo = p - pui
        return max(puo, 0)
    elif option_type == 'up-and-out put' and barrier <= K:
        puo = (
            -S0*np.exp(-q*T)*norm.cdf(-x1)
            + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
            + S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(-y1)
            - K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)*norm.cdf(-x1 + sigma*np.sqrt(T))
        )
        return max(puo, 0)

    # Down-and-in Put
    elif option_type == 'down-and-in put' and barrier < K and S0 < barrier:
        vanilla = black_scholes(S0, K, T, r, sigma, "put")
        return vanilla
    elif option_type == 'down-and-in put' and barrier > K:
        return p
    elif option_type == 'down-and-in put' and barrier < K:
        pdi = (
            -S0*np.exp(-q*T)*norm.cdf(-x1)
            + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
            + S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*(norm.cdf(y) - norm.cdf(y1))
            - K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)
              * (norm.cdf(y - sigma*np.sqrt(T))
                 - norm.cdf(y1 - sigma*np.sqrt(T)))
        )
        return pdi

    # Down-and-out Put
    elif option_type == 'down-and-out put' and barrier > K:
        return 0
    elif option_type == 'down-and-out put' and barrier < K:
        pdi = (
            -S0*np.exp(-q*T)*norm.cdf(-x1)
            + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
            + S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*(norm.cdf(y) - norm.cdf(y1))
            - K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)
              * (norm.cdf(y - sigma*np.sqrt(T))
                 - norm.cdf(y1 - sigma*np.sqrt(T)))
        )
        pdo = p - pdi
        return max(pdo, 0)

    # Fallback
    return None

def barrier_binomial_option_price(S0, K, r, q_div, T, sigma, steps,
                                  barrier_option_type, H,
                                  option_side='call', rebate=0.0):
    """
    Binomial pricing of a barrier option, tracking whether the barrier is hit.
    Incorporates continuous dividend yield q_div.
    """
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    m = math.exp((r - q_div) * dt)
    p_up = (m - d) / (u - d)

    # Determine barrier direction
    barrier_direction = "up" if barrier_option_type.lower().startswith("up") else "down"
    
    def intrinsic(S):
        return max(S - K, 0) if option_side.lower() == 'call' else max(K - S, 0)

    is_knock_out = barrier_option_type.lower().endswith("out")
    is_knock_in  = barrier_option_type.lower().endswith("in")

    memo = {}
    def f(i, j, bh):
        key = (i, j, bh)
        if key in memo:
            return memo[key]
        
        S = S0 * (u**j) * (d**(i-j))
        t = i * dt
        
        if i == steps:
            if is_knock_out:
                val = rebate if bh else intrinsic(S)
            else:
                val = intrinsic(S) if bh else 0.0
            memo[key] = val
            return val
        
        if is_knock_out and bh:
            val = rebate * math.exp(-r * (T - t))
            memo[key] = val
            return val

        S_up = S0 * (u**(j+1)) * (d**((i+1)-(j+1)))
        S_down = S0 * (u**j) * (d**((i+1)-j))

        def barrier_hit(S_new, current_bh):
            if current_bh:
                return True
            return S_new >= H if barrier_direction == "up" else S_new <= H

        new_bh_up   = barrier_hit(S_up, bh)
        new_bh_down = barrier_hit(S_down, bh)

        if is_knock_out and new_bh_up:
            val_up = rebate * math.exp(-r*(T-(t+dt)))
        else:
            val_up = f(i+1, j+1, new_bh_up)
        
        if is_knock_out and new_bh_down:
            val_down = rebate * math.exp(-r*(T-(t+dt)))
        else:
            val_down = f(i+1, j, new_bh_down)

        val = disc * (p_up * val_up + (1 - p_up) * val_down)
        memo[key] = val
        return val

    return f(0, 0, False)

def adaptive_barrier_binomial(
    S0, K, r, q, T, sigma,
    coarse_steps, fine_steps,
    barrier, barrier_option_type, option_side,
    rebate=0.0, fine_region=0.1
):
    # Define critical region around the barrier for using fine steps
    critical_region = (barrier * (1 - fine_region), barrier * (1 + fine_region))

    dt_fine = T / fine_steps
    dt_coarse = T / coarse_steps

    u_fine = np.exp(sigma * np.sqrt(dt_fine))
    d_fine = 1 / u_fine
    disc_fine = np.exp(-r * dt_fine)
    p_fine = (np.exp((r - q) * dt_fine) - d_fine) / (u_fine - d_fine)

    u_coarse = np.exp(sigma * np.sqrt(dt_coarse))
    d_coarse = 1 / u_coarse
    disc_coarse = np.exp(-r * dt_coarse)
    p_coarse = (np.exp((r - q) * dt_coarse) - d_coarse) / (u_coarse - d_coarse)

    memo = {}

    def intrinsic(S):
        return max(S - K, 0) if option_side == 'call' else max(K - S, 0)

    def adaptive_f(S, t, bh):
        if t >= T:
            if "out" in barrier_option_type:
                return rebate if bh else intrinsic(S)
            elif "in" in barrier_option_type:
                return intrinsic(S) if bh else 0
            else:
                return intrinsic(S)

        is_fine = critical_region[0] <= S <= critical_region[1]
        dt = dt_fine if is_fine else dt_coarse

        # Snap dt to maturity if needed
        if t + dt > T:
            dt = T - t
            u = np.exp(sigma * np.sqrt(dt))
            d = 1 / u
            disc = np.exp(-r * dt)
            p = (np.exp((r - q) * dt) - d) / (u - d)
        else:
            u, d = (u_fine, d_fine) if is_fine else (u_coarse, d_coarse)
            disc = disc_fine if is_fine else disc_coarse
            p = p_fine if is_fine else p_coarse

        barrier_hit = bh or (S >= barrier if "up" in barrier_option_type else S <= barrier)

        S_up, S_down = S * u, S * d
        key = (round(S, 4), round(t, 6), barrier_hit)
        if key in memo:
            return memo[key]

        val_up = adaptive_f(S_up, t + dt, barrier_hit)
        val_down = adaptive_f(S_down, t + dt, barrier_hit)
        val = disc * (p * val_up + (1 - p) * val_down)

        memo[key] = val
        return val

    return adaptive_f(S0, 0, False)

def combine_barrier_and_side(barrier_option_type, side):
    # e.g. "Up-and-Out" + "call" -> "up-and-out call"
    return f"{barrier_option_type.lower().replace(' ', '-')} {side.lower()}"

# -------------------- Adaptive Tree Visualization Function --------------------

def plot_adaptive_binomial_tree(S0, sigma, T, coarse_steps, fine_steps, barrier, fine_region):
    critical_region = (barrier * (1 - fine_region), barrier * (1 + fine_region))

    dt_coarse = T / coarse_steps if coarse_steps > 0 else T
    dt_fine = T / fine_steps if fine_steps > 0 else T

    nodes = []
    edges_x, edges_y = [], []

    def add_node(S, t, parent_t=None, parent_S=None):
        if t > T:
            return
        nodes.append((t, S))
        if parent_t is not None and parent_S is not None:
            edges_x.extend([parent_t, t, None])
            edges_y.extend([parent_S, S, None])
        if t == T:
            return

        is_fine = critical_region[0] <= S <= critical_region[1]
        dt = dt_fine if is_fine else dt_coarse
        if t + dt > T:
            dt = T - t

        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u

        add_node(S * u, t + dt, t, S)
        add_node(S * d, t + dt, t, S)

    add_node(S0, 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edges_x, y=edges_y,
        mode='lines',
        line=dict(color='gray', width=1),
        hoverinfo='none'
    ))
    xs, ys = zip(*nodes)
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode='markers',
        marker=dict(color='blue', size=5),
        name='Nodes'
    ))
    fig.add_trace(go.Scatter(
        x=[0, T],
        y=[barrier, barrier],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Barrier'
    ))
    fig.update_layout(
        title="Adaptive Binomial Tree (Snapped to Maturity)",
        xaxis_title="Time",
        yaxis_title="Asset Price",
        template="simple_white"
    )
    return fig

# -------------------- Streamlit Application --------------------

st.title("Adaptive Mesh Refinement for Barrier Options & Tree Visualization")

# Use tabs to separate functionalities
tab1, tab2 = st.tabs(["Pricing & Error Analysis", "Adaptive Binomial Tree Visualization"])

# --------- Tab 1: Pricing & Error Analysis ---------
with tab1:
    st.header("Option Pricing and Error Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Parameters")
        S0 = st.number_input("Initial Price S0", value=100.0, min_value=0.0)
        K = st.number_input("Strike Price K", value=100.0, min_value=0.0)
        r = st.number_input("Risk-free Rate r", value=0.05, min_value=0.0)
        q = st.number_input("Dividend Yield q", value=0.0, min_value=0.0)
        T = st.number_input("Time to Maturity T", value=1.0, min_value=0.0)
        sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.0)
        barrier = st.number_input("Barrier Level", value=110.0, min_value=0.0)
        option_side = st.selectbox("Option Side", ["call", "put"])
        barrier_type = st.selectbox("Barrier Type", ["Up-and-Out", "Down-and-Out", "Up-and-In", "Down-and-In"])
    with col2:
        st.subheader("AMR Parameters")
        coarse_steps = st.number_input("Coarse Steps", value=20, min_value=1)
        fine_steps = st.number_input("Fine Steps (around barrier)", value=400, min_value=1)
        fine_region = st.slider("Fine Region Width (%) around Barrier", 0.01, 0.3, 0.1)

    if st.button("Calculate Option Prices"):
        start = time.time()
        adaptive_price = adaptive_barrier_binomial(S0, K, r, q, T, sigma, coarse_steps, fine_steps, barrier, barrier_type.lower(), option_side, 0.0, fine_region)
        adaptive_time = time.time() - start

        start = time.time()
        regular_price = barrier_binomial_option_price(S0, K, r, q, T, sigma, coarse_steps, barrier_type.lower(), barrier, option_side, 0.0)
        regular_time = time.time() - start

        cf_type = combine_barrier_and_side(barrier_type, option_side)
        an_price = barrier_option_price(S0, K, T, r, q, sigma, barrier, cf_type)

        st.subheader("Results Comparison")
        st.write(f"Adaptive Binomial Price: **{adaptive_price:.4f}** (Time: {adaptive_time:.4f} s)")
        st.write(f"Regular Binomial Price: **{regular_price:.4f}** (Time: {regular_time:.4f} s)")
        st.write(f"Analytical Price: **{an_price:.4f}**")
        st.markdown("""
        - **Adaptive method** should yield more accurate results near barriers.
        - Compare computational times and errors.
        - Test various mesh refinements to balance efficiency and accuracy.
        """)

    st.subheader("Error vs. Stock Price (S0)")
    S0_min = st.number_input("Minimum S0", value=50.0, step=1.0, key='min_s0')
    S0_max = st.number_input("Maximum S0", value=150.0, step=1.0, key='max_s0')
    S0_step = st.number_input("Increment for S0", value=5.0, step=1.0, key='step_s0')

    if st.button("Generate Error Plot"):
        S0_values = np.arange(S0_min, S0_max + S0_step, S0_step)
        adaptive_errors = []
        regular_errors = []

        for s in S0_values:
            adaptive_price = adaptive_barrier_binomial(s, K, r, q, T, sigma,
                                                       coarse_steps, fine_steps,
                                                       barrier, barrier_type.lower(),
                                                       option_side, 0.0, fine_region)
            regular_price = barrier_binomial_option_price(s, K, r, q, T, sigma,
                                                          coarse_steps, barrier_type.lower(),
                                                          barrier, option_side, 0.0)
            cf_type = combine_barrier_and_side(barrier_type, option_side)
            an_price = barrier_option_price(s, K, T, r, q, sigma, barrier, cf_type)

            if an_price is not None:
                adaptive_err = np.abs(adaptive_price - an_price)
                regular_err = np.abs(regular_price - an_price)
            else:
                adaptive_err = np.nan
                regular_err = np.nan

            adaptive_errors.append(adaptive_err)
            regular_errors.append(regular_err)

        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(
            x=S0_values, y=adaptive_errors,
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=6),
            name='Adaptive Binomial'
        ))
        fig_err.add_trace(go.Scatter(
            x=S0_values, y=regular_errors,
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=6),
            name='Regular Binomial'
        ))
        fig_err.update_layout(
            title="Error vs. Stock Price (S0)",
            xaxis_title="Stock Price (S0)",
            yaxis_title="Error",
            template="simple_white"
        )
        st.plotly_chart(fig_err, use_container_width=True)

# --------- Tab 2: Adaptive Binomial Tree Visualization ---------
with tab2:
    st.header("Adaptive Binomial Tree Visualization")

    col1, col2 = st.columns(2)
    with col1:
        S0_tree = st.number_input("Initial Stock Price (S0)", value=100.0, key="tree_S0")
        sigma_tree = st.number_input("Volatility (σ)", value=0.2, key="tree_sigma")
        T_tree = st.number_input("Time to Maturity (T)", value=1.0, key="tree_T")
    with col2:
        coarse_steps_tree = st.number_input("Coarse Steps", value=5, min_value=1, key="tree_coarse")
        fine_steps_tree = st.number_input("Fine Steps", value=20, min_value=1, key="tree_fine")
        barrier_tree = st.number_input("Barrier Level", value=110.0, key="tree_barrier")
        fine_region_tree = st.slider("Fine Region (%) around Barrier", 0.01, 0.3, 0.1, key="tree_fine_region")

    if st.button("Plot Adaptive Binomial Tree"):
        fig_tree = plot_adaptive_binomial_tree(S0_tree, sigma_tree, T_tree, coarse_steps_tree, fine_steps_tree, barrier_tree, fine_region_tree)
        st.plotly_chart(fig_tree, use_container_width=True)

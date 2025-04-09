
# import streamlit as st
# import numpy as np
# import plotly.graph_objects as go
# from scipy.stats import norm
# from scipy.interpolate import interp1d
# from scipy.linalg import lu_factor, lu_solve
# import time
# import pandas as pd
# import matplotlib.pyplot as plt

# def calc_d1(S0, K, r, q, sigma, T):
#     return (np.log(S0 / K) + (r - q + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))

# def calc_d2(S0, K, r, q, sigma, T):
#     return calc_d1(S0, K, r, q, sigma, T) - sigma * np.sqrt(T)

# def calc_c(S0, K, r, q, sigma, T):
#     d1 = calc_d1(S0, K, r, q, sigma, T)
#     d2 = calc_d2(S0, K, r, q, sigma, T)
#     return (S0 * np.exp(-q*T)*norm.cdf(d1)
#             - K * np.exp(-r*T)*norm.cdf(d2))

# def calc_p(S0, K, r, q, sigma, T):
#     d1 = calc_d1(S0, K, r, q, sigma, T)
#     d2 = calc_d2(S0, K, r, q, sigma, T)
#     return (K * np.exp(-r*T)*norm.cdf(-d2)
#             - S0 * np.exp(-q*T)*norm.cdf(-d1))

# def calc_lambda(r, q, sigma):
#     # λ = (r - q + σ²/2) / σ²
#     return (r - q + 0.5 * sigma**2) / (sigma**2)

# def calc_y(barrier, S0, K, T, sigma, r, q):
#     """
#     y = [ln(barrier^2/(S0*K)) / (sigma*sqrt(T))] + λ * sigma * sqrt(T)
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
#     d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)

#     if option_type == "Call":
#         price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
#         return price
#     elif option_type == "Put":
#         price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
#         return price


# # ------------------------------
# # 2) Main barrier pricing function
# # ------------------------------

# def barrier_option_price(S0, K, T, r, q, sigma, barrier, option_type):
#     """
#     Returns the price of a barrier option (various knock-in/out types).
#     Matches standard formulas from texts like Hull, with care to keep
#     exponents and sign conventions correct.
#     """
#     x1 = calc_x1(S0, barrier, T, sigma, r, q)
#     y1 = calc_y1(S0, barrier, T, sigma, r, q)
#     c = calc_c(S0, K, r, q, sigma, T)
#     p = calc_p(S0, K, r, q, sigma, T)
#     lam = calc_lambda(r, q, sigma)
#     y  = calc_y(barrier, S0, K, T, sigma, r, q)

#     # --------------------------------
#     # Down-and-in Call
#     # --------------------------------
    
#     if option_type == 'down-and-in call' and barrier <= K and S0 <= barrier:
#         vanilla = black_scholes(S0, K, T, r, sigma, "Call")
#         return vanilla
    
#     elif option_type == 'down-and-in call' and barrier <= K:
#         # cdi, for barrier <= K
#         cdi = (S0 * np.exp(-q*T) * (barrier/S0)**(2*lam) * norm.cdf(y)
#                - K * np.exp(-r*T) * (barrier/S0)**(2*lam - 2)
#                  * norm.cdf(y - sigma*np.sqrt(T)))
#         return cdi

#     elif option_type == 'down-and-in call' and barrier >= K:
#         # cdi = c - cdo. So we compute cdo from the standard expression
#         # cdo = ...
#         # Then cdi = c - cdo
#         term1 = S0*np.exp(-q*T)*norm.cdf(x1)
#         term2 = K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
#         term3 = S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(y1)
#         term4 = K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)*norm.cdf(y1 - sigma*np.sqrt(T))
#         cdo   = term1 - term2 - term3 + term4
#         if cdo < 0:
#             cdo = 0
#             cdi   = c - cdo
#             return cdi
#         else:
#             cdi = c - cdo
#             return cdi

#     # --------------------------------
#     # Down-and-out Call
#     # --------------------------------
#     elif option_type == 'down-and-out call' and barrier <= K:

#         cdi = (S0 * np.exp(-q*T) * (barrier/S0)**(2*lam) * norm.cdf(y)
#             - K * np.exp(-r*T) * (barrier/S0)**(2*lam - 2)
#                 * norm.cdf(y - sigma*np.sqrt(T)))
#         cdo = c - cdi
#         if cdo > 0:
#             return cdo
#         else:
#             return 0

#     elif option_type == 'down-and-out call' and barrier >= K:
#         # This is the “If barrier > K” formula for the down-and-out call
#         term1 = S0 * np.exp(-q*T)*norm.cdf(x1)
#         term2 = K  * np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
#         term3 = S0 * np.exp(-q*T)*((barrier/S0)**(2*lam))*norm.cdf(y1)
#         term4 = K  * np.exp(-r*T)*((barrier/S0)**(2*lam - 2))*norm.cdf(y1 - sigma*np.sqrt(T))
#         cdo   = term1 - term2 - term3 + term4
        
#         if cdo < 0:
#             return 0
#         else:
#             return cdo

#     # --------------------------------
#     # Up-and-in Call
#     # --------------------------------
#     elif option_type == 'up-and-in call' and barrier > K:
#         # Standard up-and-in call for barrier > K
#         cui = (S0*np.exp(-q*T)*norm.cdf(x1)
#                - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
#                - S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*(norm.cdf(-y) - norm.cdf(-y1))
#                + K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)
#                  * (norm.cdf(-y + sigma*np.sqrt(T))
#                     - norm.cdf(-y1 + sigma*np.sqrt(T))))
#         return cui

#     elif option_type == 'up-and-in call' and barrier <= K:
#         # If barrier is below K, the up-and-in call is effectively the same as c
#         # or 0, depending on your setup.  Typically if barrier < S0 < K,
#         # the option knocks in only if S0 goes above barrier.  If you are sure
#         # you want to treat it as simply c, do so here:
#         return c

#     # --------------------------------
#     # Up-and-out Call
#     # --------------------------------
#     elif option_type == 'up-and-out call' and barrier <= K:
#         # If the barrier barrier <= K is below the current spot,
#         # often up-and-out call is worthless if it is truly "up" barrier?
#         return 0.0

#     elif option_type == 'up-and-out call' and barrier > K:
#         cui = (S0*np.exp(-q*T)*norm.cdf(x1)
#                - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
#                - S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*(norm.cdf(-y) - norm.cdf(-y1))
#                + K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)
#                  * (norm.cdf(-y + sigma*np.sqrt(T))
#                     - norm.cdf(-y1 + sigma*np.sqrt(T))))
#         cuo = c - cui
#         if cuo > 0:
#             return cuo
#         else:
#             return 0
        

#     # --------------------------------
#     # Up-and-in Put
#     # --------------------------------
#     elif option_type == 'up-and-in put' and barrier >= K and barrier <= S0:
#         pui = black_scholes(S0,K,T,r,sigma,"Put")
#         return pui
#     elif option_type == 'up-and-in put' and barrier >= K:
#         pui = (-S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(-y)
#                + K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)
#                  * norm.cdf(-y + sigma*np.sqrt(T)))
#         return pui
    
#         # --------------------------------
#     elif option_type == 'up-and-in put' and barrier <= K:
#         puo = (
#             -S0*np.exp(-q*T)*norm.cdf(-x1)
#             + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
#             + S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(-y1)
#             - K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)*norm.cdf(-y1 + sigma*np.sqrt(T))
#         )
#         if puo < 0:
#             puo = 0
#             pui = black_scholes(S0,K,T,r,sigma,"Put")
#             return pui
#         else:
#             pui = black_scholes(S0,K,T,r,sigma,"Put") - puo
        
#         return pui
    
#     elif option_type == 'up-and-in put' and barrier <= K:
#         # up-and-in put is the difference p - up-and-out put
#         # but for the simplified logic, we can just return p if the barrier is < K
#         return p

#     # --------------------------------
#     # Up-and-out Put
#     # --------------------------------
#     elif option_type == 'up-and-out put' and barrier >= K:
#         # puo = p - pui
#         pui = (-S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(-y)
#                + K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)
#                  * norm.cdf(-y + sigma*np.sqrt(T)))
#         if pui > 0:
#             puo = p - pui
#             return puo
#         else:
#             pui = 0
#             puo = p - pui
#             return puo

#     elif option_type == 'up-and-out put' and barrier <= K:
#         # Standard formula for barrier <= K
#         puo = (
#             -S0*np.exp(-q*T)*norm.cdf(-x1)
#             + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
#             + S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(-y1)
#             - K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)*norm.cdf(-y1 + sigma*np.sqrt(T))
#         )
#         if puo < 0:
#             puo = 0
#             return puo
#         else:
#             return puo


#     # --------------------------------
#     # Down-and-in Put
#     # --------------------------------
#     elif option_type == 'down-and-in put' and barrier < K and S0 < barrier:
#         vanilla = black_scholes(S0, K, T, r, sigma, "Put")
#         return vanilla
    
#     elif option_type == 'down-and-in put' and barrier > K:
#         # If the barrier is above K, we often treat the down-and-in put as p
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

#     # --------------------------------
#     # Down-and-out Put
#     # --------------------------------
#     elif option_type == 'down-and-out put' and barrier > K:
#         # Typically worthless if barrier > K in certain setups
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
#         if pdo > 0:
#             return pdo
#         else:
#             return 0

#     # Fallback
#     return None


# # ################################################################################
# # # 2) PDE for a Vanilla Call on [0, S_max]
# # ################################################################################

# ###############################################################################
# # 1) Vanilla PDEs (Forward Euler)
# ###############################################################################
# def forward_euler_vanilla_call(S0, K, T, r, sigma, dS, dt):
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

#     # Terminal payoff
#     V[-1, :] = np.maximum(S_grid - K, 0.0)

#     # Time array
#     t_arr = np.linspace(0, T, N + 1)

#     # Boundary conditions:
#     #   - at S=0: call is 0
#     #   - at S=S_max: call ~ S_max - K e^{-r tau}
#     for i in range(N + 1):
#         tau = T - t_arr[i]
#         V[i, 0]   = 0.0
#         V[i, -1]  = S_max - K * np.exp(-r * tau)

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


# def forward_euler_vanilla_put(S0, K, T, r, sigma, dS, dt):
#     """
#     Forward Euler PDE for a vanilla European put on [0, S_max].
#     Returns: (priceVan, S_grid, V0).
#     """
#     S_max = 2 * max(S0, K) * np.exp(r * T)
#     M = int(S_max / dS)
#     N = int(T / dt)
#     dS = S_max / M
#     dt = T / N

#     S_grid = np.linspace(0, S_max, M + 1)
#     V = np.zeros((N + 1, M + 1))

#     # Terminal payoff
#     V[-1, :] = np.maximum(K - S_grid, 0.0)

#     # Time array
#     t_arr = np.linspace(0, T, N + 1)

#     # Boundary conditions for a put:
#     #   - at S=0:  put is ~ K e^{-r tau}
#     #   - at S=S_max: put is ~ 0
#     for i in range(N + 1):
#         tau = T - t_arr[i]
#         V[i, 0]   = K * np.exp(-r * tau)  # deep in-the-money for a put
#         V[i, -1]  = 0.0

#     # PDE coefficients
#     j_arr = np.arange(M + 1)
#     a = 0.5 * dt * (sigma**2 * j_arr**2 - r * j_arr)
#     b = 1.0 - dt * (sigma**2 * j_arr**2 + r)
#     c = 0.5 * dt * (sigma**2 * j_arr**2 + r * j_arr)

#     # Forward Euler stepping
#     for n in range(N, 0, -1):
#         for j in range(1, M):
#             V[n - 1, j] = a[j] * V[n, j - 1] + b[j] * V[n, j] + c[j] * V[n, j + 1]

#     # Interpolate
#     interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
#     priceVan = float(interp_fn(S0))
#     return priceVan, S_grid, V[0, :]


# ###############################################################################
# # 2) Barrier PDEs (Forward Euler)
# ###############################################################################
# def forward_euler_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type):
#     """
#     Forward Euler for a knock-out call (either 'down-and-out' or 'up-and-out').
#     barrier_type = 'down' or 'up'
    
#     We zero out the region beyond the barrier:
#       - If 'down-and-out', zero for S <= barrier
#       - If 'up-and-out',   zero for S >= barrier
#     """
#     S_max = 2 * max(S0, K) * np.exp(r * T)
#     M = int(S_max / dS)
#     N = int(T / dt)
#     dS = S_max / M
#     dt = T / N

#     S_grid = np.linspace(0, S_max, M + 1)
#     V = np.zeros((N + 1, M + 1))

#     # Terminal payoff for a call
#     payoff = np.maximum(S_grid - K, 0.0)
#     if barrier_type == 'down':
#         # down-and-out => zero payoff for S <= barrier
#         payoff[S_grid <= barrier] = 0.0
#     else:
#         # up-and-out => zero payoff for S >= barrier
#         payoff[S_grid >= barrier] = 0.0
#     V[-1, :] = payoff

#     # Boundary conditions for a call
#     t_arr = np.linspace(0, T, N + 1)
#     for i in range(N + 1):
#         tau = T - t_arr[i]
#         V[i, 0]   = 0.0
#         V[i, -1]  = S_max - K * np.exp(-r * tau)

#     # PDE coefficients
#     j_arr = np.arange(M + 1)
#     a = 0.5 * dt * (sigma**2 * j_arr**2 - r * j_arr)
#     b = 1.0 - dt * (sigma**2 * j_arr**2 + r)
#     c = 0.5 * dt * (sigma**2 * j_arr**2 + r * j_arr)

#     # Forward Euler stepping
#     for n in range(N, 0, -1):
#         for j in range(1, M):
#             V[n - 1, j] = a[j] * V[n, j - 1] + b[j] * V[n, j] + c[j] * V[n, j + 1]

#         # Knock-out region
#         if barrier_type == 'down':
#             # zero out for S <= barrier
#             V[n - 1, S_grid <= barrier] = 0.0
#         else:
#             # zero out for S >= barrier
#             V[n - 1, S_grid >= barrier] = 0.0

#     # Price at S0
#     interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
#     priceKO = float(interp_fn(S0))
#     return priceKO, S_grid, V[0, :]


# def forward_euler_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type):
#     """
#     Forward Euler for a knock-out put (either 'down-and-out' or 'up-and-out').
#     barrier_type = 'down' or 'up'
#     """
#     S_max = 2 * max(S0, K) * np.exp(r * T)
#     M = int(S_max / dS)
#     N = int(T / dt)
#     dS = S_max / M
#     dt = T / N

#     S_grid = np.linspace(0, S_max, M + 1)
#     V = np.zeros((N + 1, M + 1))

#     # Terminal payoff for a put
#     payoff = np.maximum(K - S_grid, 0.0)
#     if barrier_type == 'down':
#         # down-and-out => zero payoff for S <= barrier
#         payoff[S_grid <= barrier] = 0.0
#     else:
#         # up-and-out => zero payoff for S >= barrier
#         payoff[S_grid >= barrier] = 0.0
#     V[-1, :] = payoff

#     # Boundary conditions for a put
#     t_arr = np.linspace(0, T, N + 1)
#     for i in range(N + 1):
#         tau = T - t_arr[i]
#         # For a put: V(0,t) ~ K e^{-r tau},  V(Smax,t) ~ 0
#         V[i, 0]   = K * np.exp(-r * tau)
#         V[i, -1]  = 0.0

#     # PDE coefficients
#     j_arr = np.arange(M + 1)
#     a = 0.5 * dt * (sigma**2 * j_arr**2 - r * j_arr)
#     b = 1.0 - dt * (sigma**2 * j_arr**2 + r)
#     c = 0.5 * dt * (sigma**2 * j_arr**2 + r * j_arr)

#     # Forward Euler stepping
#     for n in range(N, 0, -1):
#         for j in range(1, M):
#             V[n - 1, j] = a[j] * V[n, j - 1] + b[j] * V[n, j] + c[j] * V[n, j + 1]

#         # Knock-out region
#         if barrier_type == 'down':
#             V[n - 1, S_grid <= barrier] = 0.0
#         else:
#             V[n - 1, S_grid >= barrier] = 0.0

#     # Price at S0
#     interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
#     priceKO = float(interp_fn(S0))
#     return priceKO, S_grid, V[0, :]


# ###############################################################################
# # 3) Main wrapper: forward_euler(...)
# ###############################################################################
# def forward_euler(S0, K, T, r, q, sigma, dS, dt, barrier, option_type):
#     """
#     Main wrapper for forward Euler barrier options.
#     We handle the 'knock-out' case directly by zeroing out the barrier region.
#     We handle the 'knock-in' case via:
#          knock_in = vanilla - knock_out
#     """
#     # ---------------------------
#     # A) DOWN-AND-OUT CALL
#     # ---------------------------
#     if option_type == "down-and-out call":
#         return forward_euler_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down')

#     # ---------------------------
#     # B) DOWN-AND-IN CALL
#     #    = vanilla call - down-and-out call
#     # ---------------------------
#     elif option_type == "down-and-in call":
#         # 1) Price of down-and-out call
#         priceDOC, Sg_DO, PDE_DO = forward_euler_knock_out_call(
#             S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down'
#         )
#         # 2) Price of vanilla call
#         priceVan, Sg_van, PDE_van = forward_euler_vanilla_call(S0, K, T, r, sigma, dS, dt)
#         # 3) In-out parity
#         priceDin = priceVan - priceDOC
#         PDE_din  = PDE_van - PDE_DO
#         return priceDin, Sg_van, PDE_din

#     # ---------------------------
#     # C) DOWN-AND-OUT PUT
#     # ---------------------------
#     elif option_type == "down-and-out put":
#         return forward_euler_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down')

#     # ---------------------------
#     # D) DOWN-AND-IN PUT
#     #    = vanilla put - down-and-out put
#     # ---------------------------
#     elif option_type == "down-and-in put":
#         # 1) Price of down-and-out put
#         priceDOP, Sg_DO, PDE_DO = forward_euler_knock_out_put(
#             S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down'
#         )
#         # 2) Price of vanilla put
#         priceVan, Sg_van, PDE_van = forward_euler_vanilla_put(S0, K, T, r, sigma, dS, dt)
#         # 3) In-out parity
#         priceDin = priceVan - priceDOP
#         PDE_din  = PDE_van - PDE_DO
#         return priceDin, Sg_van, PDE_din

#     # ---------------------------
#     # E) UP-AND-OUT CALL
#     # ---------------------------
#     elif option_type == "up-and-out call":
#         return forward_euler_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up')

#     # ---------------------------
#     # F) UP-AND-IN CALL
#     #    = vanilla call - up-and-out call
#     # ---------------------------
#     elif option_type == "up-and-in call":
#         priceUOC, Sg_UO, PDE_UO = forward_euler_knock_out_call(
#             S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up'
#         )
#         priceVan, Sg_van, PDE_van = forward_euler_vanilla_call(S0, K, T, r, sigma, dS, dt)
#         priceUIC = priceVan - priceUOC
#         PDE_uic  = PDE_van - PDE_UO
#         return priceUIC, Sg_van, PDE_uic

#     # ---------------------------
#     # G) UP-AND-OUT PUT
#     # ---------------------------
#     elif option_type == "up-and-out put":
#         return forward_euler_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up')

#     # ---------------------------
#     # H) UP-AND-IN PUT
#     #    = vanilla put - up-and-out put
#     # ---------------------------
#     elif option_type == "up-and-in put":
#         priceUOP, Sg_UO, PDE_UO = forward_euler_knock_out_put(
#             S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up'
#         )
#         priceVan, Sg_van, PDE_van = forward_euler_vanilla_put(S0, K, T, r, sigma, dS, dt)
#         priceUIP = priceVan - priceUOP
#         PDE_uip  = PDE_van - PDE_UO
#         Sg = Sg_van-Sg_UO
#         return priceUIP, Sg_van, PDE_uip

#     return None



# ###############################################################################
# # 1) Vanilla Backward Euler (Call / Put)
# ###############################################################################
# def backward_euler_vanilla_call(S0, K, T, r, sigma, dS, dt):
#     """
#     Backward Euler PDE for a vanilla European call on [0, S_max].
#     Returns: (priceVan, S_grid, V_at_t0).
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
#     V[-1, :] = np.maximum(S_grid - K, 0.0)

#     # 3) PDE coefficients for the implicit scheme
#     j_arr = np.arange(M + 1)
#     A_ = -0.5 * dt * (sigma**2 * j_arr**2 - r * j_arr)
#     B_ =  1.0 + dt * (sigma**2 * j_arr**2 + r)
#     C_ = -0.5 * dt * (sigma**2 * j_arr**2 + r * j_arr)

#     # Tridiagonal matrix for j=1,...,M-1
#     main_diag = B_[1:M]
#     lower_diag = A_[2:M]     # subdiagonal
#     upper_diag = C_[1:M-1]   # superdiagonal
#     T_mat = np.diag(main_diag)
#     if M - 2 > 0:
#         T_mat += np.diag(lower_diag, k=-1) + np.diag(upper_diag, k=1)
#     else:
#         T_mat = T_mat.reshape((1, 1))

#     # 4) Time-stepping from n=N down to 1
#     t_arr = np.linspace(0, T, N + 1)
#     for n in range(N, 0, -1):
#         tau = T - t_arr[n - 1]
#         # Boundary conditions at time level (n-1)
#         V[n - 1, 0]   = 0.0
#         V[n - 1, -1]  = S_max - K * np.exp(-r * tau)

#         # Right-hand side from V^n
#         rhs = V[n, 1:M].copy()
#         # Adjust for known boundaries
#         rhs[0]   -= A_[1]     * V[n - 1, 0]
#         rhs[-1]  -= C_[M - 1] * V[n - 1, -1]

#         # Solve the linear system
#         V[n - 1, 1:M] = np.linalg.solve(T_mat, rhs)

#     # 5) Interpolate to get price at S0
#     interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
#     price = float(interp_fn(S0))
#     return price, S_grid, V[0, :]


# def backward_euler_vanilla_put(S0, K, T, r, sigma, dS, dt):
#     """
#     Backward Euler PDE for a vanilla European put on [0, S_max].
#     Returns: (priceVan, S_grid, V_at_t0).
#     """
#     S_max = 2 * max(S0, K) * np.exp(r * T)
#     M = int(S_max / dS)
#     N = int(T / dt)
#     dS = S_max / M
#     dt = T / N

#     S_grid = np.linspace(0, S_max, M + 1)
#     V = np.zeros((N + 1, M + 1))

#     # Terminal payoff
#     V[-1, :] = np.maximum(K - S_grid, 0.0)

#     # PDE coefficients
#     j_arr = np.arange(M + 1)
#     A_ = -0.5 * dt * (sigma**2 * j_arr**2 - r * j_arr)
#     B_ =  1.0 + dt * (sigma**2 * j_arr**2 + r)
#     C_ = -0.5 * dt * (sigma**2 * j_arr**2 + r * j_arr)

#     main_diag = B_[1:M]
#     lower_diag = A_[2:M]
#     upper_diag = C_[1:M-1]
#     T_mat = np.diag(main_diag)
#     if M - 2 > 0:
#         T_mat += np.diag(lower_diag, k=-1) + np.diag(upper_diag, k=1)
#     else:
#         T_mat = T_mat.reshape((1, 1))

#     # Time-stepping
#     t_arr = np.linspace(0, T, N + 1)
#     for n in range(N, 0, -1):
#         tau = T - t_arr[n - 1]
#         # Put boundaries:
#         V[n - 1, 0]   = K * np.exp(-r * tau)
#         V[n - 1, -1]  = 0.0

#         rhs = V[n, 1:M].copy()
#         rhs[0]   -= A_[1]     * V[n - 1, 0]
#         rhs[-1]  -= C_[M - 1] * V[n - 1, -1]

#         V[n - 1, 1:M] = np.linalg.solve(T_mat, rhs)

#     interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
#     price = float(interp_fn(S0))
#     return price, S_grid, V[0, :]


# ###############################################################################
# # 2) Knock-Out (Call / Put) with Backward Euler
# ###############################################################################
# def backward_euler_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type):
#     """
#     Backward Euler for a knock-out call:
#       barrier_type = 'down' => zero out S <= barrier
#       barrier_type = 'up'   => zero out S >= barrier
#     """
#     S_max = 2 * max(S0, K) * np.exp(r * T)
#     M = int(S_max / dS)
#     N = int(T / dt)
#     dS = S_max / M
#     dt = T / N
    
#     eps = 1e-12  # tolerance threshold

#     S_grid = np.linspace(0, S_max, M + 1)
#     V = np.zeros((N + 1, M + 1))

#     # Terminal payoff for call
#     payoff = np.maximum(S_grid - K, 0.0)
#     if barrier_type == 'down':
#         payoff[S_grid <= barrier] = 0.0
#     else:
#         payoff[S_grid >= barrier] = 0.0
#     V[-1, :] = payoff

#     # PDE coefficients
#     j_arr = np.arange(M + 1)
#     A_ = -0.5 * dt * (sigma**2 * j_arr**2 - r * j_arr)
#     B_ =  1.0 + dt * (sigma**2 * j_arr**2 + r)
#     C_ = -0.5 * dt * (sigma**2 * j_arr**2 + r * j_arr)

#     main_diag = B_[1:M]
#     lower_diag = A_[2:M]
#     upper_diag = C_[1:M-1]
#     T_mat = np.diag(main_diag)
#     if M - 2 > 0:
#         T_mat += np.diag(lower_diag, k=-1) + np.diag(upper_diag, k=1)
#     else:
#         T_mat = T_mat.reshape((1, 1))

#     # Time-stepping
#     t_arr = np.linspace(0, T, N + 1)
#     for n in range(N, 0, -1):
#         tau = T - t_arr[n - 1]
#         V[n - 1, 0]   = 0.0
#         V[n - 1, -1]  = S_max - K * np.exp(-r * tau)

#         rhs = V[n, 1:M].copy()
#         rhs[0]   -= A_[1]     * V[n - 1, 0]
#         rhs[-1]  -= C_[M - 1] * V[n - 1, -1]

#         V[n - 1, 1:M] = np.linalg.solve(T_mat, rhs)

#         # Knock out region
#         if barrier_type == 'down':
#             V[n - 1, S_grid <= barrier] = 0.0
#         else:
#             V[n - 1, S_grid >= barrier] = 0.0
            
#         V[n - 1, :] = np.where(V[n - 1, :] < eps, 0.0, V[n - 1, :])

#     interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
#     price_ko = float(interp_fn(S0))
#     return price_ko, S_grid, V[0, :]


# def backward_euler_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type):
#     """
#     Backward Euler for a knock-out put:
#       barrier_type = 'down' => zero out S <= barrier
#       barrier_type = 'up'   => zero out S >= barrier
#     """
#     S_max = 2 * max(S0, K) * np.exp(r * T)
#     M = int(S_max / dS)
#     N = int(T / dt)
#     dS = S_max / M
#     dt = T / N
    
#     eps = 1e-12

#     S_grid = np.linspace(0, S_max, M + 1)
#     V = np.zeros((N + 1, M + 1))

#     # Terminal payoff for put
#     payoff = np.maximum(K - S_grid, 0.0)
#     if barrier_type == 'down':
#         payoff[S_grid <= barrier] = 0.0
#     else:
#         payoff[S_grid >= barrier] = 0.0
#     V[-1, :] = payoff

#     # PDE coefficients
#     j_arr = np.arange(M + 1)
#     A_ = -0.5 * dt * (sigma**2 * j_arr**2 - r * j_arr)
#     B_ =  1.0 + dt * (sigma**2 * j_arr**2 + r)
#     C_ = -0.5 * dt * (sigma**2 * j_arr**2 + r * j_arr)

#     main_diag = B_[1:M]
#     lower_diag = A_[2:M]
#     upper_diag = C_[1:M-1]
#     T_mat = np.diag(main_diag)
#     if M - 2 > 0:
#         T_mat += np.diag(lower_diag, k=-1) + np.diag(upper_diag, k=1)
#     else:
#         T_mat = T_mat.reshape((1, 1))

#     # Time-stepping
#     t_arr = np.linspace(0, T, N + 1)
#     for n in range(N, 0, -1):
#         tau = T - t_arr[n - 1]
#         V[n - 1, 0]   = K * np.exp(-r * tau)
#         V[n - 1, -1]  = 0.0

#         rhs = V[n, 1:M].copy()
#         rhs[0]   -= A_[1]     * V[n - 1, 0]
#         rhs[-1]  -= C_[M - 1] * V[n - 1, -1]

#         V[n - 1, 1:M] = np.linalg.solve(T_mat, rhs)

#         # Knock out region
#         if barrier_type == 'down':
#             V[n - 1, S_grid <= barrier] = 0.0
#         else:
#             V[n - 1, S_grid >= barrier] = 0.0
        
#         V[n - 1, :] = np.where(V[n - 1, :] < eps, 0.0, V[n - 1, :])


#     interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
#     price_ko = float(interp_fn(S0))
#     return price_ko, S_grid, V[0, :]


# ###############################################################################
# # 3) Main Backward Euler Barrier Wrapper
# ###############################################################################
# def backward_euler(S0, K, T, r, q, sigma, dS, dt, barrier, option_type):
#     """
#     Main wrapper for backward Euler pricing of barrier options.
#     We implement the 'knock-out' PDE directly and use in-out parity:
#         knock_in = vanilla - knock_out
#     to get the knock-in price.
#     """
#     # ---------------------------
#     # A) DOWN-AND-OUT CALL
#     # ---------------------------
#     if option_type == "down-and-out call":
#         return backward_euler_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down')

#     # B) DOWN-AND-IN CALL = vanilla call - down-and-out call
#     elif option_type == "down-and-in call":
#         priceDOC, Sg_DO, PDE_DO = backward_euler_knock_out_call(
#             S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down'
#         )
#         priceVan, Sg_van, PDE_van = backward_euler_vanilla_call(S0, K, T, r, sigma, dS, dt)
#         priceDin = priceVan - priceDOC
#         PDE_din  = PDE_van - PDE_DO
#         return priceDin, Sg_van, PDE_din

#     # C) DOWN-AND-OUT PUT
#     elif option_type == "down-and-out put":
#         return backward_euler_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down')

#     # D) DOWN-AND-IN PUT = vanilla put - down-and-out put
#     elif option_type == "down-and-in put":
#         priceDOP, Sg_DO, PDE_DO = backward_euler_knock_out_put(
#             S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down'
#         )
#         priceVan, Sg_van, PDE_van = backward_euler_vanilla_put(S0, K, T, r, sigma, dS, dt)
#         priceDip = priceVan - priceDOP
#         PDE_dip  = PDE_van - PDE_DO
#         return priceDip, Sg_van, PDE_dip

#     # E) UP-AND-OUT CALL
#     elif option_type == "up-and-out call":
#         return backward_euler_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up')

#     # F) UP-AND-IN CALL = vanilla call - up-and-out call
#     elif option_type == "up-and-in call":
#         priceUOC, Sg_UO, PDE_UO = backward_euler_knock_out_call(
#             S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up'
#         )
#         priceVan, Sg_van, PDE_van = backward_euler_vanilla_call(S0, K, T, r, sigma, dS, dt)
#         priceUic = priceVan - priceUOC
#         PDE_uic  = PDE_van - PDE_UO
#         return priceUic, Sg_van, PDE_uic

#     # G) UP-AND-OUT PUT
#     elif option_type == "up-and-out put":
#         return backward_euler_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up')

#     # H) UP-AND-IN PUT = vanilla put - up-and-out put
#     elif option_type == "up-and-in put":
#         priceUOP, Sg_UO, PDE_UO = backward_euler_knock_out_put(
#             S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up'
#         )
#         priceVan, Sg_van, PDE_van = backward_euler_vanilla_put(S0, K, T, r, sigma, dS, dt)
#         priceUip = priceVan - priceUOP
#         PDE_uip  = PDE_van - PDE_UO
#         return priceUip, Sg_van, PDE_uip

#     # If the option_type is not recognized, return None
#     return None

# def crank_nicolson_vanilla_call(S0, K, T, r, sigma, dS, dt):
#     """
#     Crank–Nicolson PDE for a vanilla European call on [0, S_max].
#     Returns: (price, S_grid, V_at_t0).
#     """
#     eps = 1e-12  # define the error threshold
#     S_max = 2 * max(S0, K) * np.exp(r * T)
#     M = int(S_max / dS)
#     N = int(T / dt)
#     dS = S_max / M  # adjust
#     dt = T / N      # adjust

#     S_grid = np.linspace(0, S_max, M + 1)
#     V = np.zeros((N + 1, M + 1))
#     # Terminal payoff:
#     V[-1, :] = np.maximum(S_grid - K, 0.0)
    
#     # Precompute Crank–Nicolson coefficients for j = 0,...,M.
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
    
#     # Time-stepping (backward in time)
#     t_arr = np.linspace(0, T, N + 1)
#     for n in range(N, 0, -1):
#         tau = T - t_arr[n - 1]
#         # Boundary conditions for a call:
#         V[n - 1, 0]   = 0.0
#         V[n - 1, -1]  = S_max - K * np.exp(-r * tau)
        
#         # Build right-hand side for interior nodes j = 1,...,M-1.
#         # Using the explicit part:
#         #   rhs[j-1] = a[j]*V[n, j-1] + (1 - b[j])*V[n, j] + c[j]*V[n, j+1]
#         rhs = a[1:M] * V[n, 0:M-1] + (1 - b[1:M]) * V[n, 1:M] + c[1:M] * V[n, 2:M+1]
#         # Adjust for known boundary values:
#         # For j = 1 (leftmost interior): add a[1]*V[n-1,0] (V[n-1,0] is already set)
#         rhs[0]   += a[1] * V[n - 1, 0]
#         # For j = M-1: add c[M-1]*V[n-1,-1]
#         rhs[-1]  += c[M - 1] * V[n - 1, -1]
        
#         # Solve for interior nodes:
#         V[n - 1, 1:M] = np.linalg.solve(LHS, rhs)
#         V[n - 1, :] = np.where(np.abs(V[n - 1, :]) < eps, 0.0, V[n - 1, :])
    
#     # Interpolate to get the price at S0:
#     interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
#     price = float(interp_fn(S0))
#     return price, S_grid, V[0, :]


# def crank_nicolson_vanilla_put(S0, K, T, r, sigma, dS, dt):
#     """
#     Crank–Nicolson PDE for a vanilla European put on [0, S_max].
#     Returns: (price, S_grid, V_at_t0).
#     """
#     eps = 1e-12  # define the error threshold
#     S_max = 2 * max(S0, K) * np.exp(r * T)
#     M = int(S_max / dS)
#     N = int(T / dt)
#     dS = S_max / M
#     dt = T / N

#     S_grid = np.linspace(0, S_max, M + 1)
#     V = np.zeros((N + 1, M + 1))
#     # Terminal payoff:
#     V[-1, :] = np.maximum(K - S_grid, 0.0)
    
#     j_arr = np.arange(M + 1)
#     a = 0.25 * dt * (sigma**2 * j_arr**2 - r * j_arr)
#     b = 0.5  * dt * (sigma**2 * j_arr**2 + r)
#     c = 0.25 * dt * (sigma**2 * j_arr**2 + r * j_arr)
    
#     main_diag = 1 + b[1:M]
#     lower_diag = -a[2:M]
#     upper_diag = -c[1:M-1]
#     LHS = np.diag(main_diag)
#     if M - 2 > 0:
#         LHS += np.diag(lower_diag, k=-1) + np.diag(upper_diag, k=1)
    
#     t_arr = np.linspace(0, T, N + 1)
#     for n in range(N, 0, -1):
#         tau = T - t_arr[n - 1]
#         # Boundary conditions for a put:
#         V[n - 1, 0]   = K * np.exp(-r * tau)
#         V[n - 1, -1]  = 0.0
        
#         rhs = a[1:M] * V[n, 0:M-1] + (1 - b[1:M]) * V[n, 1:M] + c[1:M] * V[n, 2:M+1]
#         rhs[0]   += a[1] * V[n - 1, 0]
#         rhs[-1]  += c[M - 1] * V[n - 1, -1]
        
#         V[n - 1, 1:M] = np.linalg.solve(LHS, rhs)
#         V[n - 1, :] = np.where(np.abs(V[n - 1, :]) < eps, 0.0, V[n - 1, :])
    
#     interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
#     price = float(interp_fn(S0))
#     return price, S_grid, V[0, :]


# ###############################################################################
# # 2) Barrier Option Pricing using Crank–Nicolson
# ###############################################################################
# def crank_nicolson_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type):
#     """
#     Crank–Nicolson for a knock–out call (either 'down-and-out' or 'up-and-out').
#     barrier_type: 'down' or 'up'
    
#     The terminal payoff is set to zero in the barrier region:
#       - For 'down-and-out': zero for S <= barrier.
#       - For 'up-and-out':   zero for S >= barrier.
#     """
#     eps = 1e-12  # define the error threshold
#     S_max = 2 * max(S0, K) * np.exp(r * T)
#     M = int(S_max / dS)
#     N = int(T / dt)
#     dS = S_max / M
#     dt = T / N

#     S_grid = np.linspace(0, S_max, M + 1)
#     # Define terminal payoff and apply knockout condition:
#     payoff = np.maximum(S_grid - K, 0.0)
#     if barrier_type == 'down':
#         payoff[S_grid <= barrier] = 0.0
#     else:  # 'up'
#         payoff[S_grid >= barrier] = 0.0
#     V = np.zeros((N + 1, M + 1))
#     V[-1, :] = payoff

#     j_arr = np.arange(M + 1)
#     a = 0.25 * dt * (sigma**2 * j_arr**2 - r * j_arr)
#     b = 0.5  * dt * (sigma**2 * j_arr**2 + r)
#     c = 0.25 * dt * (sigma**2 * j_arr**2 + r * j_arr)
    
#     main_diag = 1 + b[1:M]
#     lower_diag = -a[2:M]
#     upper_diag = -c[1:M-1]
#     LHS = np.diag(main_diag)
#     if M - 2 > 0:
#         LHS += np.diag(lower_diag, k=-1) + np.diag(upper_diag, k=1)
    
#     t_arr = np.linspace(0, T, N + 1)
#     for n in range(N, 0, -1):
#         tau = T - t_arr[n - 1]
#         # Boundary conditions for a call:
#         V[n - 1, 0]   = 0.0
#         V[n - 1, -1]  = S_max - K * np.exp(-r * tau)
        
#         rhs = a[1:M] * V[n, 0:M-1] + (1 - b[1:M]) * V[n, 1:M] + c[1:M] * V[n, 2:M+1]
#         rhs[0]   += a[1] * V[n - 1, 0]
#         rhs[-1]  += c[M - 1] * V[n - 1, -1]
        
#         V[n - 1, 1:M] = np.linalg.solve(LHS, rhs)
#         V[n - 1, :] = np.where(np.abs(V[n - 1, :]) < eps, 0.0, V[n - 1, :])
#         # Enforce barrier condition at this time level:
#         if barrier_type == 'down':
#             V[n - 1, S_grid <= barrier] = 0.0
#         else:  # 'up'
#             V[n - 1, S_grid >= barrier] = 0.0
    
#     interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
#     price_ko = float(interp_fn(S0))
#     return price_ko, S_grid, V[0, :]


# def crank_nicolson_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type):
#     """
#     Crank–Nicolson for a knock–out put (either 'down-and-out' or 'up-and-out').
#     barrier_type: 'down' or 'up'
#     """
#     eps = 1e-12  # define the error threshold
#     S_max = 2 * max(S0, K) * np.exp(r * T)
#     M = int(S_max / dS)
#     N = int(T / dt)
#     dS = S_max / M
#     dt = T / N

#     S_grid = np.linspace(0, S_max, M + 1)
#     payoff = np.maximum(K - S_grid, 0.0)
#     if barrier_type == 'down':
#         payoff[S_grid <= barrier] = 0.0
#     else:
#         payoff[S_grid >= barrier] = 0.0
#     V = np.zeros((N + 1, M + 1))
#     V[-1, :] = payoff

#     j_arr = np.arange(M + 1)
#     a = 0.25 * dt * (sigma**2 * j_arr**2 - r * j_arr)
#     b = 0.5  * dt * (sigma**2 * j_arr**2 + r)
#     c = 0.25 * dt * (sigma**2 * j_arr**2 + r * j_arr)
    
#     main_diag = 1 + b[1:M]
#     lower_diag = -a[2:M]
#     upper_diag = -c[1:M-1]
#     LHS = np.diag(main_diag)
#     if M - 2 > 0:
#         LHS += np.diag(lower_diag, k=-1) + np.diag(upper_diag, k=1)
    
#     t_arr = np.linspace(0, T, N + 1)
#     for n in range(N, 0, -1):
#         tau = T - t_arr[n - 1]
#         # Boundary conditions for a put:
#         V[n - 1, 0]   = K * np.exp(-r * tau)
#         V[n - 1, -1]  = 0.0
        
#         rhs = a[1:M] * V[n, 0:M-1] + (1 - b[1:M]) * V[n, 1:M] + c[1:M] * V[n, 2:M+1]
#         rhs[0]   += a[1] * V[n - 1, 0]
#         rhs[-1]  += c[M - 1] * V[n - 1, -1]
        
#         V[n - 1, 1:M] = np.linalg.solve(LHS, rhs)
#         V[n - 1, :] = np.where(np.abs(V[n - 1, :]) < eps, 0.0, V[n - 1, :])
#         # Enforce barrier condition:
#         if barrier_type == 'down':
#             V[n - 1, S_grid <= barrier] = 0.0
#         else:
#             V[n - 1, S_grid >= barrier] = 0.0
    
#     interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
#     price_ko = float(interp_fn(S0))
#     return price_ko, S_grid, V[0, :]


# ###############################################################################
# # 3) Main Crank–Nicolson Barrier Wrapper
# ###############################################################################
# def crank_nicolson(S0, K, T, r, q, sigma, dS, dt, barrier, option_type):
#     """
#     Main wrapper for Crank–Nicolson pricing of barrier options.
#     Knock–in options are obtained via in–out parity:
#          knock_in = vanilla - knock_out.
#     option_type must be one of:
#       "down-and-out call", "down-and-in call", "down-and-out put", "down-and-in put",
#       "up-and-out call",   "up-and-in call",   "up-and-out put",   "up-and-in put".
#     """
#     if option_type == "down-and-out call":
#         return crank_nicolson_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down')
#     elif option_type == "down-and-in call":
#         price_doc, Sg, V_doc = crank_nicolson_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down')
#         price_van, Sg, V_van = crank_nicolson_vanilla_call(S0, K, T, r, sigma, dS, dt)
#         price_din = price_van - price_doc
#         V_din = V_van - V_doc
#         return price_din, Sg, V_din
#     elif option_type == "down-and-out put":
#         return crank_nicolson_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down')
#     elif option_type == "down-and-in put":
#         price_dop, Sg, V_dop = crank_nicolson_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down')
#         price_van, Sg, V_van = crank_nicolson_vanilla_put(S0, K, T, r, sigma, dS, dt)
#         price_din = price_van - price_dop
#         V_din = V_van - V_dop
#         return price_din, Sg, V_din
#     elif option_type == "up-and-out call":
#         return crank_nicolson_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up')
#     elif option_type == "up-and-in call":
#         price_uoc, Sg, V_uoc = crank_nicolson_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up')
#         price_van, Sg, V_van = crank_nicolson_vanilla_call(S0, K, T, r, sigma, dS, dt)
#         price_uic = price_van - price_uoc
#         V_uic = V_van - V_uoc
#         return price_uic, Sg, V_uic
#     elif option_type == "up-and-out put":
#         return crank_nicolson_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up')
#     elif option_type == "up-and-in put":
#         price_uop, Sg, V_uop = crank_nicolson_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up')
#         price_van, Sg, V_van = crank_nicolson_vanilla_put(S0, K, T, r, sigma, dS, dt)
#         price_uip = price_van - price_uop
#         V_uip = V_van - V_uop
#         return price_uip, Sg, V_uip
#     return None
# ################################################################################
# # 5) The Streamlit app
# ################################################################################
# # def app():
# #     # st.title("Barrier Options: PDE vs Analytical Barrier Formula")

# #     # # Sidebar inputs
# #     # S0 = st.sidebar.number_input("Spot Price (S0)", value=100.0, step=1.0)
# #     # K  = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0)
# #     # T  = st.sidebar.number_input("Time to Maturity (T)", value=1.0, step=0.00001)
# #     # r  = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)
# #     # q  = st.sidebar.number_input("Dividend Yield (q)", value=0.00, step=0.01)
# #     # sigma = st.sidebar.number_input("Volatility (sigma)", value=0.2, step=0.01)
# #     # barrier = st.sidebar.number_input("Barrier", value=80.0, step=1.0)
# #     # dS = st.sidebar.number_input("Space Step (dS)", value=1.0, step=0.1)
# #     # dt = st.sidebar.number_input("Time Step (dt)", value=0.001, step=0.001)
# #     # option_type = st.sidebar.selectbox(
# #     # "Option Type",
# #     # [
# #     #     "down-and-in call",
# #     #     "down-and-out call",
# #     #     "down-and-in put",
# #     #     "down-and-out put",
# #     #     "up-and-in call",
# #     #     "up-and-out call",
# #     #     "up-and-in put",
# #     #     "up-and-out put",
# #     # ])

# #     st.set_page_config(page_title="Numerical Scheme comparisons", layout="wide")
# #     st.title("Comparison of Forward/Backward/Crank–Nicolson Methods")

# #     # Sidebar for user inputs
# #     st.sidebar.header("Option & FD Parameters")
# #     option_type = st.sidebar.selectbox(
# #         "Option Type",
# #     [
# #         "down-and-in call",
# #         "down-and-out call",
# #         "down-and-in put",
# #         "down-and-out put",
# #         "up-and-in call",
# #         "up-and-out call",
# #         "up-and-in put",
# #         "up-and-out put",
# #     ])
# #     K          = st.sidebar.number_input("Strike (K)", value=100.0, step=1.0)
# #     T          = st.sidebar.number_input("Maturity (T, in years)", value=1.0, step=0.1)
# #     r          = st.sidebar.number_input("Risk-free rate (r)", value=0.05, step=0.01)
# #     q          = st.sidebar.number_input("Dividen-yield (q)", value=0.00, step=0.01)
# #     sigma      = st.sidebar.number_input("Volatility (sigma)", value=0.2, step=0.01)
# #     barrier    = st.sidebar.number_input("Barrier Level", value = 80.0, step = 0.1)

# #     st.sidebar.header("Range of Spot Prices")
# #     S_min      = st.sidebar.number_input("Minimum Spot (S)", value=80.0, step=1.0)
# #     S_max      = st.sidebar.number_input("Maximum Spot (S)", value=120.0, step=1.0)
# #     S_step     = st.sidebar.number_input("Spot increment", value=5.0, step=1.0)

# #     st.sidebar.header("FD Mesh Choices")
# #     # Possibly separate dt/dS for each scheme if you wish
# #     dt_explicit = st.sidebar.number_input("dt (Explicit)", value=0.0001, step=0.0001, format="%.6f")
# #     dS_explicit = st.sidebar.number_input("dS (Explicit)", value=1.0, step=0.1)

# #     dt_implicit = st.sidebar.number_input("dt (Implicit)", value=0.0001, step=0.0001, format="%.6f")
# #     dS_implicit = st.sidebar.number_input("dS (Implicit)", value=1.0, step=0.1)

# #     dt_CN       = st.sidebar.number_input("dt (Crank–Nicolson)", value=0.0001, step=0.0001, format="%.3f")
# #     dS_CN       = st.sidebar.number_input("dS (Crank–Nicolson)", value=1.0, step=0.1)

# #     # Make a list to store table rows
# #     rows = []

# #     # Iterate over the requested spot prices
# #     spots = np.arange(S_min, S_max + 0.1, S_step)
# #     err_FE_list = []
# #     err_BE_list = []
# #     err_CN_list = []
# #     if st.button("Run Scheme Comparison"):
# #         for S0 in spots:
# #             # -----------------------------------------------------
# #             #   1) True / Analytical Price
# #             # -----------------------------------------------------
# #             true_price = max(barrier_option_price(S0, K, T, r, q, sigma, barrier, option_type), 0.00000000001)

# #             # -----------------------------------------------------
# #             #   2) Forward Euler (Explicit)
# #             # -----------------------------------------------------
# #             t0 = time.perf_counter()
# #             FE_value, S_grid_FE, FE_prices = forward_euler(S0, K, T, r, q, sigma, dS_explicit, dt_explicit, barrier, option_type)
                                             
# #             time_FE  = time.perf_counter() - t0

# #             err_FE = np.abs(FE_value - true_price)

# #             err_FE_list.append(err_FE)
# #             accuracy_FE = 0.0
# #             if true_price != 0:
# #                 accuracy_FE = 100 * (1 - (err_FE / true_price))
                

# #             # -----------------------------------------------------
# #             #   3) Backward Euler (Implicit)
# #             # -----------------------------------------------------
# #             t0 = time.perf_counter()
# #             BE_value, S_grid_BE, BE_prices = backward_euler(S0, K, T, r, q, sigma, dS_implicit, dt_implicit, barrier, option_type)
# #             time_BE  = time.perf_counter() - t0

# #             err_BE = np.abs(BE_value - true_price)

# #             err_BE_list.append(err_BE)
# #             accuracy_BE = 0.0
# #             if true_price != 0:
# #                 accuracy_BE = 100 * (1 - (err_BE / true_price))

# #             # -----------------------------------------------------
# #             #   4) Crank–Nicolson
# #             # -----------------------------------------------------
# #             t0 = time.perf_counter()
# #             CN_value, S_grid_CN, CN_prices = crank_nicolson(S0, K, T, r, q, sigma, dS_CN, dt_CN, barrier, option_type)
# #             time_CN = time.perf_counter() - t0

# #             err_CN = np.abs(CN_value - true_price)

# #             err_CN_list.append(err_CN)
# #             accuracy_CN = 0.0
# #             if true_price != 0:
# #                 accuracy_CN = 100 * (1 - (err_CN / true_price))

# #             # -----------------------------------------------------
# #             #   5) Prepare row
# #             # -----------------------------------------------------
# #             row = {
# #                 "Spot": f"{S0:.2f}",
# #                 "True Value": f"{true_price:.4f}",
                
# #                 "Exp Value": f"{FE_value:.4f}",
# #                 "Exp Accuracy": f"{accuracy_FE:.2f}%",
# #                 #"Exp Time (s)": f"{time_FE:.4f}",
                
# #                 "Imp Value": f"{BE_value:.4f}",
# #                 "Imp Accuracy": f"{accuracy_BE:.2f}%",
# #                 #"Imp Time (s)": f"{time_BE:.4f}",
                
# #                 "CN Value": f"{CN_value:.4f}",
# #                 "CN Accuracy": f"{accuracy_CN:.2f}%",
# #                 #"CN Time (s)": f"{time_CN:.4f}",
# #             }
# #             rows.append(row)
            
            
            

# #         # Once done, build a final DataFrame
# #         df = pd.DataFrame(rows)

# #         st.subheader("Comparison of Three Finite‐Difference Methods vs. Black–Scholes")
# #         st.table(df)

# #     rmse_FE = np.sqrt(np.mean(np.square(err_FE_list)))
# #     rmse_BE = np.sqrt(np.mean(np.square(err_BE_list)))
# #     rmse_CN = np.sqrt(np.mean(np.square(err_CN_list)))

# #     st.markdown("### Global Error Metrics")
# #     st.write(f"**RMSE (Forward Euler):** {rmse_FE:.6f}")
# #     st.write(f"**RMSE (Backward Euler):** {rmse_BE:.6f}")
# #     st.write(f"**RMSE (Crank–Nicolson):** {rmse_CN:.6f}")

# #     # --- 1) Absolute Error vs Spot Price ---
# #     st.markdown("### 1) Absolute Error vs Spot Price")

# #     fig_err = go.Figure()

# #     # Plot Forward Euler error
# #     fig_err.add_trace(
# #         go.Scatter(
# #             x=spots,
# #             y=err_FE_list,
# #             mode='lines+markers',
# #             name='Forward Euler'
# #         )
# #     )

# #     # Plot Backward Euler error
# #     fig_err.add_trace(
# #         go.Scatter(
# #             x=spots,
# #             y=err_BE_list,
# #             mode='lines+markers',
# #             name='Backward Euler'
# #         )
# #     )

# #     # Plot Crank–Nicolson error
# #     fig_err.add_trace(
# #         go.Scatter(
# #             x=spots,
# #             y=err_CN_list,
# #             mode='lines+markers',
# #             name='Crank–Nicolson'
# #         )
# #     )

# #     fig_err.update_layout(
# #         title="Absolute Error vs Spot Price",
# #         xaxis_title="Spot Price (S₀)",
# #         yaxis_title="Absolute Error",
# #         legend_title="Method",
# #         height=500
# #     )
# #     st.plotly_chart(fig_err, use_container_width=True)

# #     # ----------------------------------------------------------------------------

# #     st.markdown("### 2) Experimental Convergence Plot (RMSE vs dt)")

# #     # You can adjust or extend dt_vals as you see fit
# #     dt_vals = [0.1, 0.05, 0.02, 0.01, 0.005]
# #     rmse_FE_dt, rmse_BE_dt, rmse_CN_dt = [], [], []

# #     # For each dt, compute the RMSE across your chosen spot range
# #     for dt in dt_vals:
# #         err_FE_temp, err_BE_temp, err_CN_temp = [], [], []
# #         for S0 in spots:
# #             fe_val, _, _ = forward_euler(
# #                 S0, K, T, r, q, sigma, dS_explicit, dt, barrier, option_type
# #             )
# #             be_val, _, _ = backward_euler(
# #                 S0, K, T, r, q, sigma, dS_implicit, dt, barrier, option_type
# #             )
# #             cn_val, _, _ = crank_nicolson(
# #                 S0, K, T, r, q, sigma, dS_CN, dt, barrier, option_type
# #             )
# #             true_val = max(barrier_option_price(S0, K, T, r, q, sigma, barrier, option_type), 0)

# #             err_FE_temp.append(abs(fe_val - true_val))
# #             err_BE_temp.append(abs(be_val - true_val))
# #             err_CN_temp.append(abs(cn_val - true_val))

# #         rmse_FE_dt.append(np.sqrt(np.mean(np.square(err_FE_temp))))
# #         rmse_BE_dt.append(np.sqrt(np.mean(np.square(err_BE_temp))))
# #         rmse_CN_dt.append(np.sqrt(np.mean(np.square(err_CN_temp))))

# #     fig_conv = go.Figure()

# #     fig_conv.add_trace(
# #         go.Scatter(
# #             x=dt_vals, y=rmse_FE_dt,
# #             mode='lines+markers', name='Forward Euler'
# #         )
# #     )
# #     fig_conv.add_trace(
# #         go.Scatter(
# #             x=dt_vals, y=rmse_BE_dt,
# #             mode='lines+markers', name='Backward Euler'
# #         )
# #     )
# #     fig_conv.add_trace(
# #         go.Scatter(
# #             x=dt_vals, y=rmse_CN_dt,
# #             mode='lines+markers', name='Crank–Nicolson'
# #         )
# #     )

# #     fig_conv.update_layout(
# #         title="RMSE vs Time Step (Convergence Plot)",
# #         xaxis_type="log",    # Log scale for dt
# #         yaxis_type="log",    # Log scale for RMSE
# #         xaxis_title="Time Step (dt)",
# #         yaxis_title="RMSE",
# #         height=500
# #     )

# #     st.plotly_chart(fig_conv, use_container_width=True)

# #     # ----------------------------------------------------------------------------

# #     st.markdown("### 3) Accuracy vs Runtime Trade-off")

# #     runtime_vals = [time_FE, time_BE, time_CN]
# #     accuracy_vals = [accuracy_FE, accuracy_BE, accuracy_CN]
# #     labels = ['Forward Euler', 'Backward Euler', 'Crank–Nicolson']

# #     fig_runtime = go.Figure()

# #     # We'll plot each method's (Runtime, Accuracy) as a single point with a label
# #     for i in range(3):
# #         fig_runtime.add_trace(
# #             go.Scatter(
# #                 x=[runtime_vals[i]],
# #                 y=[accuracy_vals[i]],
# #                 mode='markers+text',
# #                 text=[labels[i]],
# #                 name=labels[i],
# #                 textposition='top center'
# #             )
# #         )

# #     fig_runtime.update_layout(
# #         title="Accuracy vs Runtime",
# #         xaxis_title="Runtime (seconds)",
# #         yaxis_title="Accuracy (%)",
# #         height=500
# #     )

# #     st.plotly_chart(fig_runtime, use_container_width=True)
# #################################################################################################################################

# # def app():
# #     st.set_page_config(page_title="Numerical Scheme comparisons", layout="wide")
# #     st.title("Comparison of Forward/Backward/Crank–Nicolson Methods")

# #     # ============================
# #     # Sidebar for user inputs
# #     # ============================
# #     st.sidebar.header("Option & FD Parameters")
# #     option_type = st.sidebar.selectbox(
# #         "Option Type",
# #         [
# #             "down-and-in call",
# #             "down-and-out call",
# #             "down-and-in put",
# #             "down-and-out put",
# #             "up-and-in call",
# #             "up-and-out call",
# #             "up-and-in put",
# #             "up-and-out put",
# #         ],
# #     )
# #     K       = st.sidebar.number_input("Strike (K)", value=100.0, step=1.0)
# #     T       = st.sidebar.number_input("Maturity (T, in years)", value=1.0, step=0.1)
# #     r       = st.sidebar.number_input("Risk-free rate (r)", value=0.05, step=0.01)
# #     q       = st.sidebar.number_input("Dividend yield (q)", value=0.00, step=0.01)
# #     sigma   = st.sidebar.number_input("Volatility (sigma)", value=0.2, step=0.01)
# #     barrier = st.sidebar.number_input("Barrier Level", value=80.0, step=0.1)

# #     st.sidebar.header("Range of Spot Prices")
# #     S_min  = st.sidebar.number_input("Minimum Spot (S)", value=80.0, step=1.0)
# #     S_max  = st.sidebar.number_input("Maximum Spot (S)", value=120.0, step=1.0)
# #     S_step = st.sidebar.number_input("Spot increment", value=5.0, step=1.0)

# #     st.sidebar.header("FD Mesh Choices")
# #     # Possibly separate dt/dS for each scheme if you wish
# #     dt_explicit = st.sidebar.number_input("dt (Explicit)", value=0.0001, step=0.0001, format="%.6f")
# #     dS_explicit = st.sidebar.number_input("dS (Explicit)", value=1.0, step=0.1)

# #     dt_implicit = st.sidebar.number_input("dt (Implicit)", value=0.001, step=0.0001, format="%.6f")
# #     dS_implicit = st.sidebar.number_input("dS (Implicit)", value=0.5, step=0.1)

# #     dt_CN       = st.sidebar.number_input("dt (Crank–Nicolson)", value=0.01, step=0.0001, format="%.3f")
# #     dS_CN       = st.sidebar.number_input("dS (Crank–Nicolson)", value=0.5, step=0.1)

# #     # ============================
# #     # 1) Table of prices vs. "true" for multiple spots
# #     # ============================
# #     rows = []
# #     spots = np.arange(S_min, S_max + 0.1, S_step)

# #     for S0 in spots:
# #         # (A) True / Analytical Price (approx) for barrier
# #         #     If you have no closed-form, treat this as a "reference" from a high-accuracy PDE or known formula
# #         true_price = max(barrier_option_price(S0, K, T, r, q, sigma, barrier, option_type), 0.00000000001)

# #         # (B) Forward Euler (Explicit)
# #         t0 = time.perf_counter()
# #         FE_value, S_grid_FE, FE_prices = forward_euler(
# #             S0, K, T, r, q, sigma, dS_explicit, dt_explicit, barrier, option_type
# #         )
# #         time_FE = time.perf_counter() - t0
# #         err_FE = abs(FE_value - true_price)
# #         accuracy_FE = 100 * (1 - err_FE / true_price) if true_price != 0 else 0

# #         # (C) Backward Euler (Implicit)
# #         t0 = time.perf_counter()
# #         BE_value, S_grid_BE, BE_prices = backward_euler(
# #             S0, K, T, r, q, sigma, dS_implicit, dt_implicit, barrier, option_type
# #         )
# #         time_BE = time.perf_counter() - t0
# #         err_BE = abs(BE_value - true_price)
# #         accuracy_BE = 100 * (1 - err_BE / true_price) if true_price != 0 else 0

# #         # (D) Crank–Nicolson
# #         t0 = time.perf_counter()
# #         CN_value, S_grid_CN, CN_prices = crank_nicolson(
# #             S0, K, T, r, q, sigma, dS_CN, dt_CN, barrier, option_type
# #         )
# #         time_CN = time.perf_counter() - t0
# #         err_CN = abs(CN_value - true_price)
# #         accuracy_CN = 100 * (1 - err_CN / true_price) if true_price != 0 else 0

# #         # (E) Prepare row
# #         row = {
# #             "Spot": f"{S0:.2f}",
# #             "True Value": f"{true_price:.4f}",
# #             "Exp Value": f"{FE_value:.4f}",
# #             "Exp Accuracy": f"{accuracy_FE:.2f}%",
# #             "Imp Value": f"{BE_value:.4f}",
# #             "Imp Accuracy": f"{accuracy_BE:.2f}%",
# #             "CN Value": f"{CN_value:.4f}",
# #             "CN Accuracy": f"{accuracy_CN:.2f}%",
# #         }
# #         rows.append(row)

# #     df = pd.DataFrame(rows)
# #     st.subheader("1) Comparison of Three Finite‐Difference Methods vs. Reference")
# #     st.table(df)

# #     # ============================
# #     # 2) Basic Convergence Test
# #     # ============================
# #     with st.expander("Convergence & Runtime Analysis"):
# #         st.markdown("""
# #         **Goal**: Test how each scheme's accuracy and runtime behave as we refine the time-step \(\Delta t\).
# #         We'll fix a single spot \(S_0\) and compare:
# #         - Price error vs. \(\Delta t\)
# #         - CPU time vs. \(\Delta t\)
# #         """)

# #         # Let user pick a single S0 for the test
# #         test_spot = st.number_input("Spot for Convergence Test", value=100.0, step=1.0)

# #         # Range of dt values for each scheme
# #         st.write("Choose a range of dt values to test (logarithmic or linear).")
# #         dt_min = st.number_input("Min dt for test", value=0.0001, format="%.6f")
# #         dt_max = st.number_input("Max dt for test", value=0.01, format="%.6f")
# #         n_steps = st.number_input("Number of dt steps", value=5, min_value=2)

# #         if st.button("Run Convergence Test"):
# #             dt_values = np.linspace(dt_min, dt_max, n_steps)
# #             # We'll keep dS fixed for each scheme (use the user-chosen ones).
# #             # If you want to vary dS as well, you can do a nested loop.

# #             true_price_test = max(barrier_option_price(test_spot, K, T, r, q, sigma, barrier, option_type), 1e-12)
# #             results_FE = []
# #             results_BE = []
# #             results_CN = []

# #             for dt_val in dt_values:
# #                 # Forward Euler
# #                 start_FE = time.perf_counter()
# #                 FE_val, _, _ = forward_euler(test_spot, K, T, r, q, sigma, dS_explicit, dt_val, barrier, option_type)
# #                 cpu_FE = time.perf_counter() - start_FE
# #                 error_FE = abs(FE_val - true_price_test)

# #                 # Backward Euler
# #                 start_BE = time.perf_counter()
# #                 BE_val, _, _ = backward_euler(test_spot, K, T, r, q, sigma, dS_implicit, dt_val, barrier, option_type)
# #                 cpu_BE = time.perf_counter() - start_BE
# #                 error_BE = abs(BE_val - true_price_test)

# #                 # Crank–Nicolson
# #                 start_CN = time.perf_counter()
# #                 CN_val, _, _ = crank_nicolson(test_spot, K, T, r, q, sigma, dS_CN, dt_val, barrier, option_type)
# #                 cpu_CN = time.perf_counter() - start_CN
# #                 error_CN = abs(CN_val - true_price_test)

# #                 results_FE.append((dt_val, error_FE, cpu_FE))
# #                 results_BE.append((dt_val, error_BE, cpu_BE))
# #                 results_CN.append((dt_val, error_CN, cpu_CN))

# #             # Convert to DataFrame for display
# #             df_FE = pd.DataFrame(results_FE, columns=["dt", "Error", "CPU_Time"])
# #             df_BE = pd.DataFrame(results_BE, columns=["dt", "Error", "CPU_Time"])
# #             df_CN = pd.DataFrame(results_CN, columns=["dt", "Error", "CPU_Time"])

# #             st.write("**Forward Euler** results:")
# #             st.table(df_FE)
# #             st.write("**Backward Euler** results:")
# #             st.table(df_BE)
# #             st.write("**Crank–Nicolson** results:")
# #             st.table(df_CN)

# #             import plotly.graph_objects as go

# #             fig_err = go.Figure()

# #             fig_err.add_trace(go.Scatter(
# #                 x=df_FE["dt"], y=df_FE["Error"],
# #                 mode='lines+markers', name='Forward Euler'
# #             ))

# #             fig_err.add_trace(go.Scatter(
# #                 x=df_BE["dt"], y=df_BE["Error"],
# #                 mode='lines+markers', name='Backward Euler'
# #             ))

# #             fig_err.add_trace(go.Scatter(
# #                 x=df_CN["dt"], y=df_CN["Error"],
# #                 mode='lines+markers', name='Crank–Nicolson'
# #             ))

# #             fig_err.update_layout(
# #                 title=f"Error vs. dt (at S₀ = {test_spot:.2f})",
# #                 xaxis_title="Time Step (dt)",
# #                 yaxis_title="Absolute Error",
# #                 legend_title="Method",
# #                 xaxis_type="log",  # Optional: log scale for dt
# #                 yaxis_type="log",
# #                 height=500
# #             )

# #             st.plotly_chart(fig_err, use_container_width=True)
            
# #             fig_time = go.Figure()

# #             fig_time.add_trace(go.Scatter(
# #                 x=df_FE["dt"], y=df_FE["CPU_Time"],
# #                 mode='lines+markers', name='Forward Euler'
# #             ))

# #             fig_time.add_trace(go.Scatter(
# #                 x=df_BE["dt"], y=df_BE["CPU_Time"],
# #                 mode='lines+markers', name='Backward Euler'
# #             ))

# #             fig_time.add_trace(go.Scatter(
# #                 x=df_CN["dt"], y=df_CN["CPU_Time"],
# #                 mode='lines+markers', name='Crank–Nicolson'
# #             ))

# #             fig_time.update_layout(
# #                 title=f"Runtime vs. dt (at S₀ = {test_spot:.2f})",
# #                 xaxis_title="Time Step (dt)",
# #                 yaxis_title="CPU Time (seconds)",
# #                 legend_title="Method",
# #                 xaxis_type="log",
# #                 height=500
# #             )

# #             st.plotly_chart(fig_time, use_container_width=True)
# ################################################################################################
# def app():
#     st.set_page_config(page_title="Numerical Scheme comparisons", layout="wide")
#     st.title("Comparison of Forward/Backward/Crank–Nicolson Methods")

#     # ========== SIDEBAR INPUTS ==========
#     st.sidebar.header("Option & Finite Difference Parameters")
#     option_type = st.sidebar.selectbox("Option Type", [
#         "down-and-in call", "down-and-out call", "down-and-in put", "down-and-out put",
#         "up-and-in call", "up-and-out call", "up-and-in put", "up-and-out put"
#     ])
#     K = st.sidebar.number_input("Strike (K)", value=100.0)
#     T = st.sidebar.number_input("Maturity (T)", value=1.0)
#     r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05)
#     q = st.sidebar.number_input("Dividend Yield (q)", value=0.0)
#     sigma = st.sidebar.number_input("Volatility (σ)", value=0.2)
#     barrier = st.sidebar.number_input("Barrier Level", value=80.0)

#     st.sidebar.header("Spot Range")
#     S_min = st.sidebar.number_input("Minimum Spot", value=80.0)
#     S_max = st.sidebar.number_input("Maximum Spot", value=120.0)
#     S_step = st.sidebar.number_input("Spot Step", value=5.0)

#     st.sidebar.header("Finite Difference Mesh")
#     dt_explicit = st.sidebar.number_input("dt (Explicit)", value=0.0001, format="%.6f")
#     dS_explicit = st.sidebar.number_input("dS (Explicit)", value=1.0)
#     dt_implicit = st.sidebar.number_input("dt (Implicit)", value=0.001, format="%.6f")
#     dS_implicit = st.sidebar.number_input("dS (Implicit)", value=0.5)
#     dt_CN = st.sidebar.number_input("dt (Crank–Nicolson)", value=0.01, format="%.3f")
#     dS_CN = st.sidebar.number_input("dS (Crank–Nicolson)", value=0.5)

#     # ========== PRICE COMPARISON TABLE ==========
#     st.subheader("1. Accuracy Table vs Analytical Value")
#     rows, spots = [], np.arange(S_min, S_max + 0.01, S_step)

#     for S0 in spots:
#         true_val = max(barrier_option_price(S0, K, T, r, q, sigma, barrier, option_type), 1e-12)

#         FE, *_ = forward_euler(S0, K, T, r, q, sigma, dS_explicit, dt_explicit, barrier, option_type)
#         BE, *_ = backward_euler(S0, K, T, r, q, sigma, dS_implicit, dt_implicit, barrier, option_type)
#         CN, *_ = crank_nicolson(S0, K, T, r, q, sigma, dS_CN, dt_CN, barrier, option_type)

#         row = {
#             "Spot": f"{S0:.2f}",
#             "True Value": f"{true_val:.4f}",
#             "FE Value": f"{FE:.4f}", "FE Accuracy": f"{100 * (1 - abs(FE - true_val) / true_val):.2f}%",
#             "BE Value": f"{BE:.4f}", "BE Accuracy": f"{100 * (1 - abs(BE - true_val) / true_val):.2f}%",
#             "CN Value": f"{CN:.4f}", "CN Accuracy": f"{100 * (1 - abs(CN - true_val) / true_val):.2f}%",
#         }
#         rows.append(row)

#     st.table(pd.DataFrame(rows))

#     # ========== CONVERGENCE + RUNTIME TEST ==========
#     with st.expander("2. Convergence & Runtime Analysis"):
#         st.markdown("Test error and runtime for decreasing time steps.")
#         S0_test = st.number_input("Test Spot (S₀)", value=100.0)
#         dt_min = st.number_input("Minimum dt", value=0.0001, format="%.6f")
#         dt_max = st.number_input("Maximum dt", value=0.01, format="%.6f")
#         n_steps = st.number_input("Steps", min_value=2, value=6)
        
#         if st.button("Run Convergence Test"):
#             dt_vals = np.logspace(np.log10(dt_min), np.log10(dt_max), n_steps)
#             errors_FE, errors_BE, errors_CN = [], [], []
#             times_FE, times_BE, times_CN = [], [], []
#             true_val = max(barrier_option_price(S0_test, K, T, r, q, sigma, barrier, option_type), 1e-12)

#             for dt in dt_vals:
#                 t0 = time.perf_counter()
#                 fe, *_ = forward_euler(S0_test, K, T, r, q, sigma, dS_explicit, dt, barrier, option_type)
#                 times_FE.append(time.perf_counter() - t0)
#                 errors_FE.append(abs(fe - true_val))

#                 t0 = time.perf_counter()
#                 be, *_ = backward_euler(S0_test, K, T, r, q, sigma, dS_implicit, dt, barrier, option_type)
#                 times_BE.append(time.perf_counter() - t0)
#                 errors_BE.append(abs(be - true_val))

#                 t0 = time.perf_counter()
#                 cn, *_ = crank_nicolson(S0_test, K, T, r, q, sigma, dS_CN, dt, barrier, option_type)
#                 times_CN.append(time.perf_counter() - t0)
#                 errors_CN.append(abs(cn - true_val))

#             # === Plot 1: Error vs dt (Log-Log)
#             fig_err = go.Figure()
#             fig_err.add_trace(go.Scatter(x=dt_vals, y=errors_FE, mode='lines+markers', name="Forward Euler"))
#             fig_err.add_trace(go.Scatter(x=dt_vals, y=errors_BE, mode='lines+markers', name="Backward Euler"))
#             fig_err.add_trace(go.Scatter(x=dt_vals, y=errors_CN, mode='lines+markers', name="Crank–Nicolson"))
#             fig_err.update_layout(
#                 title=f"Error vs dt (S₀ = {S0_test})", xaxis_type="log", yaxis_type="log",
#                 xaxis_title="dt", yaxis_title="Absolute Error", height=500
#             )
#             st.plotly_chart(fig_err, use_container_width=True)

#             # === Plot 2: Runtime vs dt
#             fig_time = go.Figure()
#             fig_time.add_trace(go.Scatter(x=dt_vals, y=times_FE, mode='lines+markers', name="Forward Euler"))
#             fig_time.add_trace(go.Scatter(x=dt_vals, y=times_BE, mode='lines+markers', name="Backward Euler"))
#             fig_time.add_trace(go.Scatter(x=dt_vals, y=times_CN, mode='lines+markers', name="Crank–Nicolson"))
#             fig_time.update_layout(
#                 title=f"Runtime vs dt (S₀ = {S0_test})", xaxis_type="log",
#                 xaxis_title="dt", yaxis_title="CPU Time (s)", height=500
#             )
#             st.plotly_chart(fig_time, use_container_width=True)


# if __name__ == "__main__":
#     app()


# # def app():
# #     st.set_page_config(page_title="Numerical Scheme comparisons", layout="wide")
# #     st.title("Comparison of Forward/Backward/Crank–Nicolson Methods")

# #     # ========== SIDEBAR INPUTS ==========
# #     st.sidebar.header("Option & Finite Difference Parameters")
# #     option_type = st.sidebar.selectbox("Option Type", [
# #         "down-and-in call", "down-and-out call", "down-and-in put", "down-and-out put",
# #         "up-and-in call", "up-and-out call", "up-and-in put", "up-and-out put"
# #     ])
# #     K = st.sidebar.number_input("Strike (K)", value=100.0)
# #     T = st.sidebar.number_input("Maturity (T)", value=1.0)
# #     r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05)
# #     q = st.sidebar.number_input("Dividend Yield (q)", value=0.0)
# #     sigma = st.sidebar.number_input("Volatility (σ)", value=0.2)
# #     barrier = st.sidebar.number_input("Barrier Level", value=80.0)

# #     st.sidebar.header("Spot Range")
# #     S_min = st.sidebar.number_input("Minimum Spot", value=80.0)
# #     S_max = st.sidebar.number_input("Maximum Spot", value=120.0)
# #     S_step = st.sidebar.number_input("Spot Step", value=5.0)

# #     st.sidebar.header("Finite Difference Mesh")
# #     dt_explicit = st.sidebar.number_input("dt (Explicit)", value=0.0001, format="%.6f")
# #     dS_explicit = st.sidebar.number_input("dS (Explicit)", value=1.0)
# #     dt_implicit = st.sidebar.number_input("dt (Implicit)", value=0.001, format="%.6f")
# #     dS_implicit = st.sidebar.number_input("dS (Implicit)", value=0.5)
# #     dt_CN = st.sidebar.number_input("dt (Crank–Nicolson)", value=0.01, format="%.3f")
# #     dS_CN = st.sidebar.number_input("dS (Crank–Nicolson)", value=0.5)

# #     # ========== PRICE COMPARISON TABLE ==========
# #     st.subheader("1. Accuracy Table vs Analytical Value")
# #     rows, spots = [], np.arange(S_min, S_max + 0.01, S_step)

# #     for S0 in spots:
# #         true_val = max(barrier_option_price(S0, K, T, r, q, sigma, barrier, option_type), 1e-12)

# #         FE, *_ = forward_euler(S0, K, T, r, q, sigma, dS_explicit, dt_explicit, barrier, option_type)
# #         BE, *_ = backward_euler(S0, K, T, r, q, sigma, dS_implicit, dt_implicit, barrier, option_type)
# #         CN, *_ = crank_nicolson(S0, K, T, r, q, sigma, dS_CN, dt_CN, barrier, option_type)

# #         row = {
# #             "Spot": f"{S0:.2f}",
# #             "True Value": f"{true_val:.4f}",
# #             "FE Value": f"{FE:.4f}", "FE Accuracy": f"{100 * (1 - abs(FE - true_val) / true_val):.2f}%",
# #             "BE Value": f"{BE:.4f}", "BE Accuracy": f"{100 * (1 - abs(BE - true_val) / true_val):.2f}%",
# #             "CN Value": f"{CN:.4f}", "CN Accuracy": f"{100 * (1 - abs(CN - true_val) / true_val):.2f}%",
# #         }
# #         rows.append(row)

# #     st.table(pd.DataFrame(rows))

# #     # ========== CONVERGENCE + RUNTIME TEST ==========
# #     with st.expander("2. Convergence & Runtime Analysis"):
# #         st.markdown("Test error and runtime for decreasing time steps.")
# #         S0_test = st.number_input("Test Spot (S₀)", value=100.0)
# #         dt_min = st.number_input("Minimum dt", value=0.0001, format="%.6f")
# #         dt_max = st.number_input("Maximum dt", value=0.01, format="%.6f")
# #         n_steps = st.number_input("Steps", min_value=2, value=6)

# #         if st.button("Run Convergence Test"):
# #             dt_vals = np.logspace(np.log10(dt_min), np.log10(dt_max), n_steps)
# #             errors_FE, errors_BE, errors_CN = [], [], []
# #             times_FE, times_BE, times_CN = [], [], []
# #             true_val = max(barrier_option_price(S0_test, K, T, r, q, sigma, barrier, option_type), 1e-12)

# #             for dt in dt_vals:
# #                 t0 = time.perf_counter()
# #                 fe, *_ = forward_euler(S0_test, K, T, r, q, sigma, dS_explicit, dt, barrier, option_type)
# #                 times_FE.append(time.perf_counter() - t0)
# #                 errors_FE.append(abs(fe - true_val))

# #                 t0 = time.perf_counter()
# #                 be, *_ = backward_euler(S0_test, K, T, r, q, sigma, dS_implicit, dt, barrier, option_type)
# #                 times_BE.append(time.perf_counter() - t0)
# #                 errors_BE.append(abs(be - true_val))

# #                 t0 = time.perf_counter()
# #                 cn, *_ = crank_nicolson(S0_test, K, T, r, q, sigma, dS_CN, dt, barrier, option_type)
# #                 times_CN.append(time.perf_counter() - t0)
# #                 errors_CN.append(abs(cn - true_val))

# #             # === Plot 1: Error vs dt (Log-Log)
# #             fig_err = go.Figure()
# #             fig_err.add_trace(go.Scatter(x=dt_vals, y=errors_FE, mode='lines+markers', name="Forward Euler"))
# #             fig_err.add_trace(go.Scatter(x=dt_vals, y=errors_BE, mode='lines+markers', name="Backward Euler"))
# #             fig_err.add_trace(go.Scatter(x=dt_vals, y=errors_CN, mode='lines+markers', name="Crank–Nicolson"))
# #             fig_err.update_layout(
# #                 title=f"Error vs dt (S₀ = {S0_test})", xaxis_type="log", yaxis_type="log",
# #                 xaxis_title="dt", yaxis_title="Absolute Error", height=500
# #             )
# #             st.plotly_chart(fig_err, use_container_width=True)

# #             # === Plot 2: Runtime vs dt
# #             fig_time = go.Figure()
# #             fig_time.add_trace(go.Scatter(x=dt_vals, y=times_FE, mode='lines+markers', name="Forward Euler"))
# #             fig_time.add_trace(go.Scatter(x=dt_vals, y=times_BE, mode='lines+markers', name="Backward Euler"))
# #             fig_time.add_trace(go.Scatter(x=dt_vals, y=times_CN, mode='lines+markers', name="Crank–Nicolson"))
# #             fig_time.update_layout(
# #                 title=f"Runtime vs dt (S₀ = {S0_test})", xaxis_type="log",
# #                 xaxis_title="dt", yaxis_title="CPU Time (s)", height=500
# #             )
# #             st.plotly_chart(fig_time, use_container_width=True)

# #             # === Plot 3: Error vs Time Step (Linear)
# #             fig_err_linear = go.Figure()
# #             fig_err_linear.add_trace(go.Scatter(x=dt_vals, y=errors_FE, mode='lines+markers', name="Forward Euler"))
# #             fig_err_linear.add_trace(go.Scatter(x=dt_vals, y=errors_BE, mode='lines+markers', name="Backward Euler"))
# #             fig_err_linear.add_trace(go.Scatter(x=dt_vals, y=errors_CN, mode='lines+markers', name="Crank–Nicolson"))
# #             fig_err_linear.update_layout(
# #                 title=f"Error vs Time Step (S₀ = {S0_test})", xaxis_title="Time Step", yaxis_title="Error", height=500
# #             )
# #             st.plotly_chart(fig_err_linear, use_container_width=True)

# #             # === Plot 4: Accuracy vs Spot Price
# #             fig_accuracy = go.Figure()
# #             fig_accuracy.add_trace(go.Scatter(x=spots, y=[100 * (1 - abs(fe - true_val) / true_val) for fe in errors_FE], mode='lines+markers', name="Forward Euler"))
# #             fig_accuracy.add_trace(go.Scatter(x=spots, y=[100 * (1 - abs(be - true_val) / true_val) for be in errors_BE], mode='lines+markers', name="Backward Euler"))
# #             fig_accuracy.add_trace(go.Scatter(x=spots, y=[100 * (1 - abs(cn - true_val) / true_val) for cn in errors_CN], mode='lines+markers', name="Crank–Nicolson"))
# #             fig_accuracy.update_layout(
# #                 title=f"Accuracy vs Spot Price (S₀)", xaxis_title="Spot Price", yaxis_title="Accuracy (%)", height=500
# #             )
# #             st.plotly_chart(fig_accuracy, use_container_width=True)

# # if __name__ == "__main__":
# #     app()


import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.interpolate import interp1d
import time
import pandas as pd

# ----------------------------
#  Analytical Pricing Section
# ----------------------------
def black_scholes_vanilla(S, K, T, r, q, sigma, option_type):
    """ Compute vanilla European call/put price using Black–Scholes. """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) + 1e-12)
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        return None

def analytic_barrier_price(S, K, T, r, q, sigma, barrier, option_type):
    """
    A simple analytical reference using in-out parity.
    For knock-out: use an approximate closed form if available.
    Here we assume that if the barrier knocks out the option, then:
        Price(knock-out) = BlackScholes_vanilla if S is far from barrier,
        and 0 if the barrier is hit.
    For knock-in, we use:
        Price(knock-in) = Price(vanilla) - Price(knock-out).
    In practice, use a full closed form if available.
    """
    vanilla = black_scholes_vanilla(S, K, T, r, q, sigma, "call" if "call" in option_type.lower() else "put")
    # For a knock-out option, as a very crude approximation,
    # set the price to zero when S is beyond barrier (depending on type)
    if "out" in option_type.lower():
        if ("down" in option_type.lower() and S <= barrier) or ("up" in option_type.lower() and S >= barrier):
            return 0.0
        else:
            # crude adjustment: discount by a factor based on barrier proximity
            factor = 0.9 if abs(S - barrier) < 10 else 1.0
            return vanilla * factor
    elif "in" in option_type.lower():
        # use in-out parity: Price(in) = vanilla - Price(out)
        # here we compute knock-out price via the crude method above
        ko = analytic_barrier_price(S, K, T, r, q, sigma, barrier, option_type.replace("in", "out"))
        return vanilla - ko
    else:
        return vanilla

# ----------------------------
# Finite Difference Helper Functions
# ----------------------------
def thomas_solver(a, b, c, d):
    """
    Thomas algorithm for a tridiagonal system.
    a: sub-diagonal (length n-1)
    b: main diagonal (length n)
    c: super-diagonal (length n-1)
    d: right-hand side (length n)
    Returns the solution vector x.
    """
    n = len(b)
    cp = np.empty(n-1)
    dp = np.empty(n)
    
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n-1):
        denom = b[i] - a[i-1] * cp[i-1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i-1] * dp[i-1]) / denom
    dp[-1] = (d[-1] - a[-1] * dp[-2]) / (b[-1] - a[-1] * cp[-1])
    
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
    return x

def setup_grid(S_min, S_max, dS, T, dt):
    S_grid = np.arange(S_min, S_max + dS, dS)
    N_t = int(np.round(T/dt))
    dt = T / N_t
    return S_grid, N_t, dt

# ----------------------------
# Finite Difference Schemes
# ----------------------------
def forward_euler_vanilla_call(S0, K, T, r, q, sigma, S_grid, N_t, dt):
    """ Forward Euler (explicit) for a European call. """
    M = len(S_grid) - 1
    V = np.zeros((N_t+1, M+1))
    
    # Terminal condition:
    V[-1, :] = np.maximum(S_grid - K, 0)
    
    # Boundary conditions: at S = S_min and S = S_max
    t_arr = np.linspace(0, T, N_t+1)
    for n in range(N_t+1):
        tau = T - t_arr[n]
        V[n, 0] = 0  # call value at S = 0
        V[n, -1] = S_grid[-1] * np.exp(-q * tau) - K * np.exp(-r * tau)
    
    # Coefficients: using central differences
    for n in range(N_t, 0, -1):
        for j in range(1, M):
            S = S_grid[j]
            dV_dS = (V[n, j+1] - V[n, j-1]) / (2 * (S_grid[1]-S_grid[0]))
            d2V_dS2 = (V[n, j+1] - 2*V[n, j] + V[n, j-1]) / ((S_grid[1]-S_grid[0])**2)
            V[n-1, j] = V[n, j] + dt * (
                0.5*sigma**2 * S**2 * d2V_dS2 +
                (r - q)*S * dV_dS - r*V[n, j]
            )
    # Interpolate at S0
    f = interp1d(S_grid, V[0, :], kind='linear', fill_value="extrapolate")
    return float(f(S0)), V[0, :]

def backward_euler_vanilla_call(S0, K, T, r, q, sigma, S_grid, N_t, dt):
    """ Backward Euler (implicit) for a European call. """
    M = len(S_grid) - 1
    V = np.zeros((N_t+1, M+1))
    V[-1, :] = np.maximum(S_grid - K, 0)
    
    # Create tri-diagonal coefficients (for interior points 1:M)
    dS = S_grid[1]-S_grid[0]
    j = np.arange(1, M)
    a = -0.5 * dt * (sigma**2 * (S_grid[j]**2)/(dS**2) - (r - q)*S_grid[j]/(2*dS))
    b = 1 + dt * (sigma**2 * (S_grid[j]**2)/(dS**2) + r)
    c = -0.5 * dt * (sigma**2 * (S_grid[j]**2)/(dS**2) + (r - q)*S_grid[j]/(2*dS))
    
    # Time stepping backward
    t_arr = np.linspace(0, T, N_t+1)
    for n in range(N_t, 0, -1):
        tau = T - t_arr[n-1]
        V[n-1, 0] = 0
        V[n-1, -1] = S_grid[-1]*np.exp(-q*tau) - K*np.exp(-r*tau)
        # Right-hand side:
        d = V[n, 1:M].copy()
        # Adjust for boundary conditions:
        d[0]   -= a[0] * V[n-1, 0]
        d[-1]  -= c[-1] * V[n-1, -1]
        # Solve tridiagonal system
        V[n-1, 1:M] = thomas_solver(a, b, c, d)
    
    f = interp1d(S_grid, V[0, :], kind='linear', fill_value="extrapolate")
    return float(f(S0)), V[0, :]

def crank_nicolson_vanilla_call(S0, K, T, r, q, sigma, S_grid, N_t, dt):
    """ Crank–Nicolson for a European call with two implicit (Rannacher) steps. """
    M = len(S_grid) - 1
    V = np.zeros((N_t+1, M+1))
    dS = S_grid[1]-S_grid[0]
    V[-1, :] = np.maximum(S_grid - K, 0)
    
    # Precompute coefficients for interior nodes:
    j = np.arange(1, M)
    a = 0.25 * dt * (sigma**2 * (S_grid[j]**2)/(dS**2) - (r - q)*S_grid[j]/(dS))
    b = -0.5 * dt * (sigma**2 * (S_grid[j]**2)/(dS**2) + r)
    c = 0.25 * dt * (sigma**2 * (S_grid[j]**2)/(dS**2) + (r - q)*S_grid[j]/(dS))
    
    # Build matrices for Crank–Nicolson: For implicit part (matrix A) and explicit part (B).
    A_diag = 1 - b
    A_sub = -a
    A_sup = -c
    B_diag = 1 + b
    B_sub = a
    B_sup = c
    
    # Pre-assemble tridiagonals (they are constant in time here)
    # Use these to solve: A * V_new = B * V_old + adjustments from BC.
    for n in range(N_t, 0, -1):
        # Set boundary conditions:
        tau = T - (n-1)*dt
        V[n-1, 0] = 0
        V[n-1, -1] = S_grid[-1]*np.exp(-q*tau) - K*np.exp(-r*tau)
        # Build RHS for interior nodes:
        d = np.zeros(M-1)
        V_old = V[n, 1:M]
        # B matrix action:
        d = B_sub * V[n, 0:M-1] + B_diag * V_old + B_sup * V[n, 2:M+1]
        # Adjust for boundary values:
        d[0]   += a[0] * V[n-1,0]
        d[-1]  += c[-1] * V[n-1,-1]
        # Solve the system A * X = d:
        X = thomas_solver(A_sub, A_diag, A_sup, d)
        V[n-1, 1:M] = X
    f = interp1d(S_grid, V[0, :], kind='linear', fill_value="extrapolate")
    return float(f(S0)), V[0, :]

# ----------------------------
# Barrier PDE via FD (using knock-out scheme)
# ----------------------------
def forward_euler_knock_out(S0, K, T, r, q, sigma, S_grid, N_t, dt, barrier, barrier_side, option_type):
    """ Forward Euler for a barrier option: zero out the knock-out region. """
    M = len(S_grid) - 1
    V = np.zeros((N_t+1, M+1))
    dS = S_grid[1]-S_grid[0]
    # Terminal condition for call/put payoff:
    if "call" in option_type.lower():
        payoff = np.maximum(S_grid - K, 0)
    else:
        payoff = np.maximum(K - S_grid, 0)
    
    # Apply barrier knock-out to terminal payoff:
    if barrier_side == "down":
        payoff[S_grid <= barrier] = 0
    else:
        payoff[S_grid >= barrier] = 0
    V[-1, :] = payoff
    
    t_arr = np.linspace(0, T, N_t+1)
    for n in range(N_t+1):
        tau = T - t_arr[n]
        # Standard BC for vanilla:
        if "call" in option_type.lower():
            V[n, 0] = 0
            V[n, -1] = S_grid[-1]*np.exp(-q*tau) - K*np.exp(-r*tau)
        else:
            V[n, 0] = K*np.exp(-r*tau)
            V[n, -1] = 0
            
    for n in range(N_t, 0, -1):
        for j in range(1, M):
            S = S_grid[j]
            dV_dS = (V[n, j+1] - V[n, j-1])/(2*dS)
            d2V_dS2 = (V[n, j+1] - 2*V[n, j] + V[n, j-1])/(dS**2)
            V[n-1, j] = V[n, j] + dt * (
                0.5 * sigma**2 * S**2 * d2V_dS2 + (r-q)*S*dV_dS - r*V[n, j]
            )
        # Enforce barrier condition:
        if barrier_side == "down":
            V[n-1, S_grid <= barrier] = 0
        else:
            V[n-1, S_grid >= barrier] = 0
    f = interp1d(S_grid, V[0, :], kind='linear', fill_value="extrapolate")
    return float(f(S0)), V[0, :]

def backward_euler_knock_out(S0, K, T, r, q, sigma, S_grid, N_t, dt, barrier, barrier_side, option_type):
    """ Backward Euler for a barrier option: zero out the barrier region at each time step. """
    M = len(S_grid) - 1
    V = np.zeros((N_t+1, M+1))
    dS = S_grid[1]-S_grid[0]
    # Terminal payoff:
    if "call" in option_type.lower():
        payoff = np.maximum(S_grid - K, 0)
    else:
        payoff = np.maximum(K - S_grid, 0)
    if barrier_side == "down":
        payoff[S_grid <= barrier] = 0
    else:
        payoff[S_grid >= barrier] = 0
    V[-1, :] = payoff
    
    t_arr = np.linspace(0, T, N_t+1)
    for n in range(N_t+1):
        tau = T - t_arr[n]
        if "call" in option_type.lower():
            V[n, 0] = 0
            V[n, -1] = S_grid[-1]*np.exp(-q*tau) - K*np.exp(-r*tau)
        else:
            V[n, 0] = K*np.exp(-r*tau)
            V[n, -1] = 0
            
    j = np.arange(1, M)
    a = -0.5 * dt * (sigma**2 * (S_grid[j]**2)/(dS**2) - (r - q)*S_grid[j]/(2*dS))
    b = 1 + dt * (sigma**2 * (S_grid[j]**2)/(dS**2) + r)
    c = -0.5 * dt * (sigma**2 * (S_grid[j]**2)/(dS**2) + (r - q)*S_grid[j]/(2*dS))
    
    for n in range(N_t, 0, -1):
        tau = T - t_arr[n-1]
        if "call" in option_type.lower():
            V[n-1, 0] = 0
            V[n-1, -1] = S_grid[-1]*np.exp(-q*tau) - K*np.exp(-r*tau)
        else:
            V[n-1, 0] = K*np.exp(-r*tau)
            V[n-1, -1] = 0
        d = V[n, 1:M].copy()
        d[0]   -= a[0] * V[n-1, 0]
        d[-1]  -= c[-1] * V[n-1, -1]
        V[n-1, 1:M] = thomas_solver(a, b, c, d)
        # Enforce barrier condition:
        if barrier_side == "down":
            V[n-1, S_grid <= barrier] = 0
        else:
            V[n-1, S_grid >= barrier] = 0
    f = interp1d(S_grid, V[0, :], kind='linear', fill_value="extrapolate")
    return float(f(S0)), V[0, :]

def crank_nicolson_knock_out(S0, K, T, r, q, sigma, S_grid, N_t, dt, barrier, barrier_side, option_type):
    """ Crank–Nicolson for a barrier option with barrier condition enforced at each step. """
    M = len(S_grid) - 1
    V = np.zeros((N_t+1, M+1))
    dS = S_grid[1]-S_grid[0]
    if "call" in option_type.lower():
        payoff = np.maximum(S_grid - K, 0)
    else:
        payoff = np.maximum(K - S_grid, 0)
    if barrier_side == "down":
        payoff[S_grid <= barrier] = 0
    else:
        payoff[S_grid >= barrier] = 0
    V[-1, :] = payoff
    
    t_arr = np.linspace(0, T, N_t+1)
    for n in range(N_t+1):
        tau = T - t_arr[n]
        if "call" in option_type.lower():
            V[n, 0] = 0
            V[n, -1] = S_grid[-1]*np.exp(-q*tau) - K*np.exp(-r*tau)
        else:
            V[n, 0] = K*np.exp(-r*tau)
            V[n, -1] = 0

    j = np.arange(1, M)
    a = 0.25 * dt * (sigma**2 * (S_grid[j]**2)/(dS**2) - (r - q)*S_grid[j]/(dS))
    b = -0.5 * dt * (sigma**2 * (S_grid[j]**2)/(dS**2) + r)
    c = 0.25 * dt * (sigma**2 * (S_grid[j]**2)/(dS**2) + (r - q)*S_grid[j]/(dS))
    
    A_diag = 1 - b
    A_sub = -a
    A_sup = -c
    B_diag = 1 + b
    B_sub = a
    B_sup = c

    for n in range(N_t, 0, -1):
        tau = T - t_arr[n-1]
        if "call" in option_type.lower():
            V[n-1, 0] = 0
            V[n-1, -1] = S_grid[-1]*np.exp(-q*tau) - K*np.exp(-r*tau)
        else:
            V[n-1, 0] = K*np.exp(-r*tau)
            V[n-1, -1] = 0
        d = B_sub * V[n, 0:M-1] + B_diag * V[n, 1:M] + B_sup * V[n, 2:M+1]
        d[0]   += a[0] * V[n-1,0]
        d[-1]  += c[-1] * V[n-1,-1]
        X = thomas_solver(A_sub, A_diag, A_sup, d)
        V[n-1, 1:M] = X
        if barrier_side == "down":
            V[n-1, S_grid <= barrier] = 0
        else:
            V[n-1, S_grid >= barrier] = 0
    f = interp1d(S_grid, V[0, :], kind='linear', fill_value="extrapolate")
    return float(f(S0)), V[0, :]

# ----------------------------
# Wrapper Functions: FD for Barrier Options using In-Out Parity
# ----------------------------
def forward_euler(S0, K, T, r, q, sigma, dS, dt, barrier, option_type):
    # Set grid: for barrier problems, choose S_min so that the barrier is on the grid.
    if "down" in option_type.lower():
        S_min = barrier * 0.8
    else:
        S_min = 0.0
    S_max = 2 * max(S0, K) * np.exp(r*T)
    S_grid, N_t, dt = setup_grid(S_min, S_max, dS, T, dt)
    
    # For knock-out options, solve directly; for knock-in, use in-out parity.
    if "out" in option_type.lower():
        side = "down" if "down" in option_type.lower() else "up"
        price, FD_curve = forward_euler_knock_out(S0, K, T, r, q, sigma, S_grid, N_t, dt, barrier, side, option_type)
        return price, S_grid, FD_curve
    elif "in" in option_type.lower():
        # Price knock-out and subtract from vanilla
        side = "down" if "down" in option_type.lower() else "up"
        priceKO, FD_curveKO = forward_euler_knock_out(S0, K, T, r, q, sigma, S_grid, N_t, dt, barrier, side, option_type.replace("in","out"))
        # Vanilla price via FD
        priceVan, FD_curveVan = forward_euler_vanilla_call(S0, K, T, r, q, sigma, S_grid, N_t, dt)
        priceIN = priceVan - priceKO
        FD_curveIN = FD_curveVan - FD_curveKO
        return priceIN, S_grid, FD_curveIN
    else:
        # Vanilla
        price, FD_curve = forward_euler_vanilla_call(S0, K, T, r, q, sigma, S_grid, N_t, dt)
        return price, S_grid, FD_curve

def backward_euler(S0, K, T, r, q, sigma, dS, dt, barrier, option_type):
    if "down" in option_type.lower():
        S_min = barrier * 0.8
    else:
        S_min = 0.0
    S_max = 2 * max(S0, K) * np.exp(r*T)
    S_grid, N_t, dt = setup_grid(S_min, S_max, dS, T, dt)
    if "out" in option_type.lower():
        side = "down" if "down" in option_type.lower() else "up"
        price, FD_curve = backward_euler_knock_out(S0, K, T, r, q, sigma, S_grid, N_t, dt, barrier, side, option_type)
        return price, S_grid, FD_curve
    elif "in" in option_type.lower():
        side = "down" if "down" in option_type.lower() else "up"
        priceKO, FD_curveKO = backward_euler_knock_out(S0, K, T, r, q, sigma, S_grid, N_t, dt, barrier, side, option_type.replace("in","out"))
        priceVan, FD_curveVan = backward_euler_vanilla_call(S0, K, T, r, q, sigma, S_grid, N_t, dt)
        priceIN = priceVan - priceKO
        FD_curveIN = FD_curveVan - FD_curveKO
        return priceIN, S_grid, FD_curveIN
    else:
        price, FD_curve = backward_euler_vanilla_call(S0, K, T, r, q, sigma, S_grid, N_t, dt)
        return price, S_grid, FD_curve

def crank_nicolson(S0, K, T, r, q, sigma, dS, dt, barrier, option_type):
    if "down" in option_type.lower():
        S_min = barrier * 0.8
    else:
        S_min = 0.0
    S_max = 2 * max(S0, K) * np.exp(r*T)
    S_grid, N_t, dt = setup_grid(S_min, S_max, dS, T, dt)
    if "out" in option_type.lower():
        side = "down" if "down" in option_type.lower() else "up"
        price, FD_curve = crank_nicolson_knock_out(S0, K, T, r, q, sigma, S_grid, N_t, dt, barrier, side, option_type)
        return price, S_grid, FD_curve
    elif "in" in option_type.lower():
        side = "down" if "down" in option_type.lower() else "up"
        priceKO, FD_curveKO = crank_nicolson_knock_out(S0, K, T, r, q, sigma, S_grid, N_t, dt, barrier, side, option_type.replace("in","out"))
        priceVan, FD_curveVan = crank_nicolson_vanilla_call(S0, K, T, r, q, sigma, S_grid, N_t, dt)
        priceIN = priceVan - priceKO
        FD_curveIN = FD_curveVan - FD_curveKO
        return priceIN, S_grid, FD_curveIN
    else:
        price, FD_curve = crank_nicolson_vanilla_call(S0, K, T, r, q, sigma, S_grid, N_t, dt)
        return price, S_grid, FD_curve

# ----------------------------
# Streamlit App Layout
# ----------------------------
def app():
    st.set_page_config(page_title="Numerical Scheme Comparisons", layout="wide")
    st.title("Comparison of Forward/Backward/Crank–Nicolson Methods")

    # Sidebar Inputs
    st.sidebar.header("Option & FD Parameters")
    option_type = st.sidebar.selectbox(
        "Option Type", [
            "down-and-in call", "down-and-out call",
            "down-and-in put", "down-and-out put",
            "up-and-in call", "up-and-out call",
            "up-and-in put", "up-and-out put"
        ]
    )
    K = st.sidebar.number_input("Strike (K)", value=100.0)
    T = st.sidebar.number_input("Maturity (T, years)", value=1.0, format="%.2f")
    r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05, format="%.3f")
    q = st.sidebar.number_input("Dividend Yield (q)", value=0.0, format="%.3f")
    sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, format="%.3f")
    barrier = st.sidebar.number_input("Barrier Level", value=80.0)

    st.sidebar.header("Spot Price Range")
    S_min_input = st.sidebar.number_input("Minimum Spot (S)", value=80.0)
    S_max_input = st.sidebar.number_input("Maximum Spot (S)", value=120.0)
    S_step = st.sidebar.number_input("Spot Step", value=5.0)

    st.sidebar.header("FD Mesh Parameters")
    dt_explicit = st.sidebar.number_input("dt (Explicit)", value=0.0001, format="%.6f")
    dS_explicit = st.sidebar.number_input("dS (Explicit)", value=1.0)
    dt_implicit = st.sidebar.number_input("dt (Implicit)", value=0.001, format="%.6f")
    dS_implicit = st.sidebar.number_input("dS (Implicit)", value=0.5)
    dt_CN = st.sidebar.number_input("dt (Crank–Nicolson)", value=0.01, format="%.3f")
    dS_CN = st.sidebar.number_input("dS (Crank–Nicolson)", value=0.5)

    # Accuracy Table vs Analytical Value
    st.subheader("1. Accuracy Table vs Analytical Value")
    rows = []
    spots = np.arange(S_min_input, S_max_input + 0.01, S_step)
    for S0 in spots:
        true_val = analytic_barrier_price(S0, K, T, r, q, sigma, barrier, option_type)
        FE, _, _ = forward_euler(S0, K, T, r, q, sigma, dS_explicit, dt_explicit, barrier, option_type)
        BE, _, _ = backward_euler(S0, K, T, r, q, sigma, dS_implicit, dt_implicit, barrier, option_type)
        CN, _, _ = crank_nicolson(S0, K, T, r, q, sigma, dS_CN, dt_CN, barrier, option_type)
        rows.append({
            "Spot": f"{S0:.2f}",
            "Analytical": f"{true_val:.4f}",
            "FE": f"{FE:.4f}",
            "FE Accuracy": f"{100*(1-abs(FE-true_val)/true_val):.2f}%" if true_val != 0 else "N/A",
            "BE": f"{BE:.4f}",
            "BE Accuracy": f"{100*(1-abs(BE-true_val)/true_val):.2f}%" if true_val != 0 else "N/A",
            "CN": f"{CN:.4f}",
            "CN Accuracy": f"{100*(1-abs(CN-true_val)/true_val):.2f}%" if true_val != 0 else "N/A",
        })
    st.table(pd.DataFrame(rows))

    # Convergence & Runtime Analysis
    with st.expander("2. Convergence & Runtime Analysis"):
        st.markdown("Run convergence tests for a fixed spot (S₀).")
        S0_test = st.number_input("Test Spot (S₀)", value=100.0)
        dt_min = st.number_input("Minimum dt", value=0.0001, format="%.6f")
        dt_max = st.number_input("Maximum dt", value=0.01, format="%.6f")
        n_steps = st.number_input("Number of dt steps", value=6, min_value=2)
        
        if st.button("Run Convergence Test"):
            dt_vals = np.logspace(np.log10(dt_min), np.log10(dt_max), n_steps)
            errors_FE, errors_BE, errors_CN = [], [], []
            times_FE, times_BE, times_CN = [], [], []
            true_val = analytic_barrier_price(S0_test, K, T, r, q, sigma, barrier, option_type)
            for dt in dt_vals:
                t0 = time.perf_counter()
                fe, _, _ = forward_euler(S0_test, K, T, r, q, sigma, dS_explicit, dt, barrier, option_type)
                times_FE.append(time.perf_counter()-t0)
                errors_FE.append(abs(fe-true_val))
                
                t0 = time.perf_counter()
                be, _, _ = backward_euler(S0_test, K, T, r, q, sigma, dS_implicit, dt, barrier, option_type)
                times_BE.append(time.perf_counter()-t0)
                errors_BE.append(abs(be-true_val))
                
                t0 = time.perf_counter()
                cn, _, _ = crank_nicolson(S0_test, K, T, r, q, sigma, dS_CN, dt, barrier, option_type)
                times_CN.append(time.perf_counter()-t0)
                errors_CN.append(abs(cn-true_val))
            
            # Plot Error vs dt (log-log)
            fig_err = go.Figure()
            fig_err.add_trace(go.Scatter(x=dt_vals, y=errors_FE, mode='lines+markers', name="Forward Euler"))
            fig_err.add_trace(go.Scatter(x=dt_vals, y=errors_BE, mode='lines+markers', name="Backward Euler"))
            fig_err.add_trace(go.Scatter(x=dt_vals, y=errors_CN, mode='lines+markers', name="Crank–Nicolson"))
            fig_err.update_layout(title=f"Error vs dt (S₀ = {S0_test})",
                                  xaxis_title="dt", yaxis_title="Absolute Error",
                                  xaxis_type="log", yaxis_type="log", height=500)
            st.plotly_chart(fig_err, use_container_width=True)
            
            # Plot Runtime vs dt
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(x=dt_vals, y=times_FE, mode='lines+markers', name="Forward Euler"))
            fig_time.add_trace(go.Scatter(x=dt_vals, y=times_BE, mode='lines+markers', name="Backward Euler"))
            fig_time.add_trace(go.Scatter(x=dt_vals, y=times_CN, mode='lines+markers', name="Crank–Nicolson"))
            fig_time.update_layout(title=f"Runtime vs dt (S₀ = {S0_test})",
                                   xaxis_title="dt", yaxis_title="CPU Time (s)",
                                   xaxis_type="log", height=500)
            st.plotly_chart(fig_time, use_container_width=True)
    
    # Plot Error vs Spot Price
    st.subheader("3. Absolute Error vs Spot Price")
    error_FE_spot, error_BE_spot, error_CN_spot = [], [], []
    true_vals = []
    for S0 in spots:
        true_val = analytic_barrier_price(S0, K, T, r, q, sigma, barrier, option_type)
        true_vals.append(true_val)
        fe, _, _ = forward_euler(S0, K, T, r, q, sigma, dS_explicit, dt_explicit, barrier, option_type)
        be, _, _ = backward_euler(S0, K, T, r, q, sigma, dS_implicit, dt_implicit, barrier, option_type)
        cn, _, _ = crank_nicolson(S0, K, T, r, q, sigma, dS_CN, dt_CN, barrier, option_type)
        error_FE_spot.append(abs(fe-true_val))
        error_BE_spot.append(abs(be-true_val))
        error_CN_spot.append(abs(cn-true_val))
    fig_spot = go.Figure()
    fig_spot.add_trace(go.Scatter(x=spots, y=error_FE_spot, mode='lines+markers', name="Forward Euler"))
    fig_spot.add_trace(go.Scatter(x=spots, y=error_BE_spot, mode='lines+markers', name="Backward Euler"))
    fig_spot.add_trace(go.Scatter(x=spots, y=error_CN_spot, mode='lines+markers', name="Crank–Nicolson"))
    fig_spot.update_layout(title="Absolute Error vs Spot Price", xaxis_title="Spot Price", yaxis_title="Absolute Error", height=500)
    st.plotly_chart(fig_spot, use_container_width=True)
    
if __name__ == "__main__":
    app()

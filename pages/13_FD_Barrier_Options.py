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

# # def black_scholes(S, K, T, r, sigma, option_type):
# #     d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
# #     d2 = d1 - sigma * np.sqrt(T)

# #     if option_type == "Call":
# #         price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
# #         return price
# #     elif option_type == "Put":
# #         price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
# #         return price


# # # ------------------------------
# # # 2) Main barrier pricing function
# # # ------------------------------

# # def barrier_option_price(S0, K, T, r, q, sigma, H, option_type):
# #     """
# #     Returns the price of a barrier option (various knock-in/out types).
# #     Matches standard formulas from texts like Hull, with care to keep
# #     exponents and sign conventions correct.
# #     """
# #     x1 = calc_x1(S0, H, T, sigma, r, q)
# #     y1 = calc_y1(S0, H, T, sigma, r, q)
# #     c = calc_c(S0, K, r, q, sigma, T)
# #     p = calc_p(S0, K, r, q, sigma, T)
# #     lam = calc_lambda(r, q, sigma)
# #     y  = calc_y(barrier, S0, K, T, sigma, r, q)

# #     # --------------------------------
# #     # Down-and-in Call
# #     # --------------------------------
    
# #     if option_type == 'down-and-in call' and barrier <= K and np.any(S0) <= barrier:
# #         vanilla = black_scholes(S0, K, T, r, sigma, "Call")
# #         return vanilla
    
# #     elif option_type == 'down-and-in call' and barrier <= K:
# #         # cdi, for H <= K
# #         cdi = (S0 * np.exp(-q*T) * (H/S0)**(2*lam) * norm.cdf(y)
# #                - K * np.exp(-r*T) * (H/S0)**(2*lam - 2)
# #                  * norm.cdf(y - sigma*np.sqrt(T)))
# #         return cdi

# #     elif option_type == 'down-and-in call' and H >= K:
# #         # cdi = c - cdo. So we compute cdo from the standard expression
# #         # cdo = ...
# #         # Then cdi = c - cdo
# #         term1 = S0*np.exp(-q*T)*norm.cdf(x1)
# #         term2 = K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
# #         term3 = S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(y1)
# #         term4 = K*np.exp(-r*T)*(H/S0)**(2*lam - 2)*norm.cdf(y1 - sigma*np.sqrt(T))
# #         cdo   = term1 - term2 - term3 + term4
# #         if cdo < 0:
# #             cdo = 0
# #             cdi   = c - cdo
# #             return cdi
# #         else:
# #             return cdi

# #     # --------------------------------
# #     # Down-and-out Call
# #     # --------------------------------
# #     elif option_type == 'down-and-out call' and H <= K:

# #         cdi = (S0 * np.exp(-q*T) * (H/S0)**(2*lam) * norm.cdf(y)
# #             - K * np.exp(-r*T) * (H/S0)**(2*lam - 2)
# #                 * norm.cdf(y - sigma*np.sqrt(T)))
# #         cdo = c - cdi
# #         if cdo > 0:
# #             return cdo
# #         else:
# #             return 0

# #     elif option_type == 'down-and-out call' and H >= K:
# #         # This is the “If H > K” formula for the down-and-out call
# #         term1 = S0 * np.exp(-q*T)*norm.cdf(x1)
# #         term2 = K  * np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
# #         term3 = S0 * np.exp(-q*T)*((H/S0)**(2*lam))*norm.cdf(y1)
# #         term4 = K  * np.exp(-r*T)*((H/S0)**(2*lam - 2))*norm.cdf(y1 - sigma*np.sqrt(T))
# #         cdo   = term1 - term2 - term3 + term4
        
# #         if cdo < 0:
# #             return 0
# #         else:
# #             return cdo

# #     # --------------------------------
# #     # Up-and-in Call
# #     # --------------------------------
# #     elif option_type == 'up-and-in call' and H > K:
# #         # Standard up-and-in call for H > K
# #         cui = (S0*np.exp(-q*T)*norm.cdf(x1)
# #                - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
# #                - S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(-y) - norm.cdf(-y1))
# #                + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
# #                  * (norm.cdf(-y + sigma*np.sqrt(T))
# #                     - norm.cdf(-y1 + sigma*np.sqrt(T))))
# #         return cui

# #     elif option_type == 'up-and-in call' and H <= K:
# #         # If barrier is below K, the up-and-in call is effectively the same as c
# #         # or 0, depending on your setup.  Typically if H < S0 < K,
# #         # the option knocks in only if S0 goes above H.  If you are sure
# #         # you want to treat it as simply c, do so here:
# #         return c

# #     # --------------------------------
# #     # Up-and-out Call
# #     # --------------------------------
# #     elif option_type == 'up-and-out call' and H <= K:
# #         # If the barrier H <= K is below the current spot,
# #         # often up-and-out call is worthless if it is truly "up" barrier?
# #         return 0.0

# #     elif option_type == 'up-and-out call' and H > K:
# #         cui = (S0*np.exp(-q*T)*norm.cdf(x1)
# #                - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
# #                - S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(-y) - norm.cdf(-y1))
# #                + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
# #                  * (norm.cdf(-y + sigma*np.sqrt(T))
# #                     - norm.cdf(-y1 + sigma*np.sqrt(T))))
# #         cuo = c - cui
# #         return cuo

# #     # --------------------------------
# #     # Up-and-in Put
# #     # --------------------------------
# #     elif option_type == 'up-and-in put' and H >= K:
# #         pui = (-S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y)
# #                + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
# #                  * norm.cdf(-y + sigma*np.sqrt(T)))
# #         return pui
    
# #         # --------------------------------
# #     elif option_type == 'up-and-in put' and H <= K:
# #         puo = (
# #             -S0*np.exp(-q*T)*norm.cdf(-x1)
# #             + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
# #             + S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y1)
# #             - K*np.exp(-r*T)*(H/S0)**(2*lam - 2)*norm.cdf(-y1 + sigma*np.sqrt(T))
# #         )
# #         if puo < 0:
# #             puo = 0
# #             pui = black_scholes(S0,K,T,r,sigma,"Put")
# #             return pui
# #         else:
# #             pui = black_scholes(S0,K,T,r,sigma,"Put") - puo
        
# #         return pui
    
# #     elif option_type == 'up-and-in put' and H <= K:
# #         # up-and-in put is the difference p - up-and-out put
# #         # but for the simplified logic, we can just return p if the barrier is < K
# #         return p

# #     # --------------------------------
# #     # Up-and-out Put
# #     # --------------------------------
# #     elif option_type == 'up-and-out put' and H >= K:
# #         # puo = p - pui
# #         pui = (-S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y)
# #                + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
# #                  * norm.cdf(-y + sigma*np.sqrt(T)))
# #         puo = p - pui
# #         return puo

# #     elif option_type == 'up-and-out put' and H <= K:
# #         # Standard formula for H <= K
# #         puo = (
# #             -S0*np.exp(-q*T)*norm.cdf(-x1)
# #             + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
# #             + S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y1)
# #             - K*np.exp(-r*T)*(H/S0)**(2*lam - 2)*norm.cdf(-y1 + sigma*np.sqrt(T))
# #         )
# #         if puo < 0:
# #             return 0
# #         else:
# #             return puo

# #     # --------------------------------
# #     # Down-and-in Put
# #     # --------------------------------
# #     elif option_type == 'down-and-in put' and H < K and S0 < H:
# #         vanilla = black_scholes(S0, K, T, r, sigma, "Put")
# #         return vanilla
    
# #     elif option_type == 'down-and-in put' and H > K:
# #         # If the barrier is above K, we often treat the down-and-in put as p
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
# #         # Typically worthless if H > K in certain setups
# #         return 0

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
# #         if pdo > 0:
# #             return pdo
# #         else:
# #             return 0

# #     # Fallback
# #     return None

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
#         return cuo

#     # --------------------------------
#     # Up-and-in Put
#     # --------------------------------
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
#         puo = p - pui
#         return puo

#     elif option_type == 'up-and-out put' and barrier <= K:
#         # Standard formula for barrier <= K
#         puo = (
#             -S0*np.exp(-q*T)*norm.cdf(-x1)
#             + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
#             + S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(-y1)
#             - K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)*norm.cdf(-y1 + sigma*np.sqrt(T))
#         )
#         if puo < 0:
#             return 0
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


# # Forward Euler finite difference method
# def forward_eulerV(S0, K, T, r, sigma, dS, dt, option_type):
#     S_max = 2*max(S0,K)*np.exp(r*T)
#     M = int(S_max / dS)
#     N = int(T / dt)
#     dS = S_max / M
#     dt = T / N


#     matval = np.zeros((M + 1, N + 1))
#     vetS = np.linspace(0, S_max, M + 1)

#     # Boundary conditions
#     if option_type == "Call":
#         matval[:, -1] = np.maximum(vetS - K, 0)
#         matval[0, :] = 0
#         matval[-1, :] = S_max - K * np.exp(-r * (T - np.linspace(0, T, N + 1)))
#     elif option_type == "Put":
#         matval[:, -1] = np.maximum(K - vetS, 0)
#         matval[0, :] = K * np.exp(-r * (T - np.linspace(0, T, N + 1)))
#         matval[-1, :] = 0

#     # Coefficients
#     a = 0.5 * dt * (sigma**2 * np.arange(M + 1) - r) * np.arange(M + 1)
#     b = 1 - dt * (sigma**2 * np.arange(M + 1)**2 + r)
#     c = 0.5 * dt * (sigma**2 * np.arange(M + 1) + r) * np.arange(M + 1)

#     # Time-stepping
#     for j in range(N, 0, -1):
#         for i in range(1, M):
#             matval[i, j - 1] = (
#                 a[i] * matval[i - 1, j]
#                 + b[i] * matval[i, j]
#                 + c[i] * matval[i + 1, j]
#             )

#     price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
#     price = price_interp(S0)

        
#     return price, matval[:,0]



# def backward_eulerV(S0, K, r, T, sigma, dS, dt, option_type):
#     # set up grid and adjust increments if necessary
#     Smax = 2*max(S0,K)*np.exp(r*T)
#     M = round(Smax / dS)
#     dS = Smax / M
#     N = round(T / dt)
#     dt = T / N
#     matval = np.zeros((M + 1, N + 1))
#     vetS = np.linspace(0, Smax, M + 1)
#     veti = np.arange(0, M + 1)
#     vetj = np.arange(0, N + 1)
    
#     # Boundary conditions
#     if option_type == "Call":
#         matval[:, -1] = np.maximum(vetS - K, 0)
#         matval[0, :] = 0
#         #matval[-1, :] = S_max - K * np.exp(-r * (T - np.linspace(0, T, N + 1)))
#         matval[-1, :] = Smax - K * np.exp(-r * dt * (N - vetj))
#     elif option_type == "Put":
#         matval[:, -1] = np.maximum(K - vetS, 0)
#         matval[0, :] = K * np.exp(-r * (T - np.linspace(0, T, N + 1)))
#         matval[-1, :] = 0
    
#     # set up the tridiagonal coefficients matrix
#     a = 0.5 * (r * dt * veti - sigma**2 * dt * (veti**2))
#     b = 1 + sigma**2 * dt * (veti**2) + r * dt
#     c = -0.5 * (r * dt * veti + sigma**2 * dt * (veti**2))
#     coeff = np.diag(a[2:M], -1) + np.diag(b[1:M]) + np.diag(c[1:M-1], 1)
#     #lu, piv = lu_factor(coeff)
    
#     if option_type == "Put":
        
#         LU, piv = lu_factor(coeff)  # Equivalent to MATLAB's [L, U] = lu(coeff)

#         # Solve the sequence of linear systems
#         aux = np.zeros(M-1)

#         for j in range(N-1, -1, -1):  # Reverse loop from N to 1
#             aux[0] = -a[1] * matval[0, j]  # Adjust indexing for Python (0-based)
    
#             # Solve L(Ux) = b using LU decomposition
#             matval[1:M, j] = lu_solve((LU, piv), matval[1:M, j+1] + aux)
         
#         price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
#         price = price_interp(S0)
        
#         return price
    
#     elif option_type == "Call":
#         LU, piv = lu_factor(coeff)  # Equivalent to MATLAB's [L, U] = lu(coeff)

#         # Solve the sequence of linear systems
#         aux = np.zeros(M-1)

#         for j in range(N-1, -1, -1):  # Reverse loop from N to 1
#             aux[M-2] = -c[M-1] * matval[M, j]  # Adjust indexing for Python (0-based)
    
#             # Solve L(Ux) = b using LU decomposition
#             matval[1:M, j] = lu_solve((LU, piv), matval[1:M, j+1] + aux)
            
#         price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
#         price = price_interp(S0)

        
#         return price
    
# def crank_nicolsonV(S0, K, r, T, sigma, dS, dt, option_type):
#     Smax = 2*max(S0,K)*np.exp(r*T)
#     M = round(Smax / dS)
#     dS = Smax / M
#     N = round(T / dt)
#     dt = T / N
#     matval = np.zeros((M+1, N+1))
#     vetS = np.linspace(0, Smax, M+1)
#     veti = np.arange(0, M+1)
#     vetj = np.arange(0, N+1)

#     if option_type == "Call":
#         matval[:, -1] = np.maximum(vetS - K, 0)
#         matval[0, :] = 0
#         #matval[-1, :] = S_max - K * np.exp(-r * (T - np.linspace(0, T, N + 1)))
#         matval[-1, :] = Smax - K * np.exp(-r * dt * (N - vetj))
#     elif option_type == "Put":
#         matval[:, -1] = np.maximum(K - vetS, 0)
#         matval[0, :] = K * np.exp(-r * (T - np.linspace(0, T, N + 1)))
#         matval[-1, :] = 0

#     # Set up the coefficients matrix
#     alpha = 0.25 * dt * (sigma**2 * (veti**2) - r * veti)
#     beta = -0.5 * dt * (sigma**2 * (veti**2) + r)
#     gamma = 0.25 * dt * (sigma**2 * (veti**2) + r * veti)

#     # Construct tridiagonal matrices
#     M1 = -np.diag(alpha[2:M], -1) + np.diag(1 - beta[1:M]) - np.diag(gamma[1:M-1], 1)
#     M2 = np.diag(alpha[2:M], -1) + np.diag(1 + beta[1:M]) + np.diag(gamma[1:M-1], 1)

#     # LU decomposition for efficient solving
#     LU, piv = lu_factor(M1)

#     # Solve the sequence of linear systems
#     lostval = np.zeros(M2.shape[1])

#     for j in range(N-1, -1, -1):
#         if len(lostval) > 1:
#             lostval[0] = alpha[1] * (matval[0, j] + matval[0, j+1])
#             lostval[-1] = gamma[-1] * (matval[-1, j] + matval[-1, j+1])
#         else:
#             lostval = lostval[0] + lostval[-1]

#         rhs = M2 @ matval[1:M, j+1] + lostval
#         matval[1:M, j] = lu_solve((LU, piv), rhs) 
    
#     price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
#     price = price_interp(S0)
                
#     return price

# # # Black-Scholes formula for analytical solution
# # def black_scholes(S, K, T, r, sigma, option_type):
# #     d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
# #     d2 = d1 - sigma * np.sqrt(T)

# #     if option_type == "Call":
# #         price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
# #         return price
# #     elif option_type == "Put":
# #         price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
# #         return price

# # Forward Euler finite difference method
# def forward_euler(S0, K, T, r, sigma, dS, dt, option_type, barrier):
#     if option_type == "down-and-out put":
#         matval, veti, M, N, vetS, r =DOPut(S0,K,r,T,barrier,dS,dt)
#     elif option_type == "up-and-out put":
#         matval, veti, M, N, vetS, r =UOPut(S0,K,r,T,barrier,dS,dt)
#     elif option_type == "up-and-out call":
#         matval, veti, M, N, vetS, r =UOCall(S0,K,r,T,barrier,dS,dt)
#     elif option_type == "down-and-out call":
#         matval, veti, M, N, vetS, r =DOCall(S0,K,r,T,barrier,dS,dt)
#     elif option_type == "down-and-in call":
#         doc, vetS, _ = forward_euler(S0, K, T, r, sigma, dS, dt, "down-and-out call", barrier)
#         if doc < 0:
#             doc = 0
#             #price = black_scholes(S0, K, T, r, sigma, "Call") - doc
#             price = forward_eulerV(S0, K, T, r, sigma, dS, dt, "Call") - doc
#             _, matval = forward_eulerV(S0, K, T, r, sigma, dS, dt, "Call")
#         else:
#             #price = black_scholes(S0, K, T, r, sigma, "Call") - doc
#             price = forward_eulerV(S0, K, T, r, sigma, dS, dt, "Call") - doc
#             _, matval = forward_eulerV(S0, K, T, r, sigma, dS, dt, "Call")
#         return price,vetS, matval[:,0]
#     elif option_type == "down-and-in put":
#         dop, _, _ = forward_euler(S0, K, T, r, sigma, dS, dt, "down-and-out put", barrier)
#         price = forward_eulerV(S0, K, T, r, sigma, dS, dt, "Put") - dop
#         return price,_,_
#     elif option_type == "up-and-in put":
#         uop, _, _ = forward_euler(S0, K, T, r, sigma, dS, dt, "up-and-out put", barrier)
#         if uop < 0:
#             uop = 0
#             price = forward_eulerV(S0, K, T, r, sigma, dS, dt, "Put") - uop
#         else: 
#             price = forward_eulerV(S0, K, T, r, sigma, dS, dt, "Put") - uop
#         return price,_,_
#     elif option_type == "up-and-in call":
#         uop, _, _ = forward_euler(S0, K, T, r, sigma, dS, dt, "up-and-out call", barrier)
#         price = forward_eulerV(S0, K, T, r, sigma, dS, dt, "Call") - uop
#         return price,_,_
#     elif option_type not in ("down-and-out put", "up-and-out put", "up-and-out call", "down-and-out call"):
#         st.error("UnboundLocalError: The variable 'veti' has not been assigned a value. This is likely due to an unsupported option type. Please select a valid option type, such as 'down-and-out put', 'up-and-out put', 'up-and-out call' or 'down-and-out call'.")
#         st.stop()
#     else:
#         pass
        
#     # Coefficients
#     a = 0.5 * dt * (((sigma**2) * veti) - r) * veti
#     b = 1 - dt * (((sigma**2) * (veti**2)) + r)
#     c = 0.5 * dt * (((sigma**2) * veti) + r) * veti

#     # Time-stepping
#     for j in range(N, 0, -1):
#         for i in range(1, M):
#             matval[i, j - 1] = (
#                 a[i] * matval[i - 1, j]
#                 + b[i] * matval[i, j]
#                 + c[i] * matval[i + 1, j]
#             )

#     price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
#     price = price_interp(S0)

        
#     return price, vetS, matval[:, 0]



# def backward_euler(S0, K, r, T, sigma, dS, dt, option_type, barrier):
#     if option_type == "down-and-out put":
#         matval, veti, M, N, vetS, r =DOPut(S0,K,r,T,barrier,dS,dt)
#     elif option_type == "up-and-out put":
#         matval, veti, M, N, vetS, r =UOPut(S0,K,r,T,barrier,dS,dt)
#     elif option_type == "up-and-out call":
#         matval, veti, M, N, vetS, r =UOCall(S0,K,r,T,barrier,dS,dt)
#     elif option_type == "down-and-out call":
#         matval, veti, M, N, vetS, r =DOCall(S0,K,r,T,barrier,dS,dt)
#     elif option_type == "down-and-in call":
#         doc, _, _ = backward_euler(S0, K, r, T, sigma, dS, dt, "down-and-out call", barrier)
#         if doc < 0:
#             doc = 0
#             price = backward_eulerV(S0, K, r, T, sigma, dS, dt, "Call") - doc
#         else:
#             price = backward_eulerV(S0, K, r, T, sigma, dS, dt, "Call") - doc
#         return price,_,_
#     elif option_type == "down-and-in put":
#         dop, _, _ = backward_euler(S0, K, r, T, sigma, dS, dt, "down-and-out put", barrier)
#         price = backward_eulerV(S0, K, r, T, sigma, dS, dt, "Put") - dop
#         return price,_,_
#     elif option_type == "up-and-in put":
#         uop, _, _ = backward_euler(S0, K, r, T, sigma, dS, dt, "up-and-out put", barrier)
#         if uop < 0:
#             uop = 0
#             price = backward_eulerV(S0, K, r, T, sigma, dS, dt, "Put") - uop
#         else:
#             price = backward_eulerV(S0, K, r, T, sigma, dS, dt, "Put") - uop 
#         return price,_,_
#     elif option_type == "up-and-in call":
#         uop, _, _ = backward_euler(S0, K, r, T, sigma, dS, dt, "up-and-out call", barrier)
#         price = backward_eulerV(S0, K, r, T, sigma, dS, dt, "Call") - uop
#         return price,_,_
#     elif option_type not in ("down-and-out put", "up-and-out put", "up-and-out call", "down-and-out call"):
#         st.error("UnboundLocalError: The variable 'veti' has not been assigned a value. This is likely due to an unsupported option type. Please select a valid option type, such as 'down-and-out put', 'up-and-out put', 'up-and-out call' or 'down-and-out call'.")
#         st.stop()     
    
#     # set up the tridiagonal coefficients matrix
#     a = 0.5 * (r * dt * veti - sigma**2 * dt * (veti**2))
#     b = 1 + sigma**2 * dt * (veti**2) + r * dt
#     c = -0.5 * (r * dt * veti + sigma**2 * dt * (veti**2))
#     coeff = np.diag(a[2:M], -1) + np.diag(b[1:M]) + np.diag(c[1:M-1], 1)
#     #lu, piv = lu_factor(coeff)
    
#     if "put" in option_type:
        
#         LU, piv = lu_factor(coeff)  # Equivalent to MATLAB's [L, U] = lu(coeff)

#         # Solve the sequence of linear systems
#         aux = np.zeros(M-1)

#         for j in range(N-1, -1, -1):  # Reverse loop from N to 1
#             aux[0] = -a[1] * matval[0, j]  # Adjust indexing for Python (0-based)
    
#             # Solve L(Ux) = b using LU decomposition
#             matval[1:M, j] = lu_solve((LU, piv), matval[1:M, j+1] + aux)
         
#         price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
#         price = price_interp(S0)
        
#         return price, vetS, matval[:, 0]
    
#     elif "call" in option_type:
#         LU, piv = lu_factor(coeff)  # Equivalent to MATLAB's [L, U] = lu(coeff)

#         # Solve the sequence of linear systems
#         aux = np.zeros(M-1)

#         for j in range(N-1, -1, -1):  # Reverse loop from N to 1
#             aux[M-2] = -c[M-1] * matval[M, j]  # Adjust indexing for Python (0-based)
    
#             # Solve L(Ux) = b using LU decomposition
#             matval[1:M, j] = lu_solve((LU, piv), matval[1:M, j+1] + aux)
            
#         price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
#         price = price_interp(S0)

        
#         return price, vetS, matval[:, 0]
    
# def crank_nicolson(S0, K, r, T, sigma, dS, dt, option_type, barrier):

#     if option_type == "down-and-out put":
#         matval, veti, M, N, vetS, r =DOPut(S0,K,r,T,barrier,dS,dt)
#     elif option_type == "up-and-out put":
#         matval, veti, M, N, vetS, r =UOPut(S0,K,r,T,barrier,dS,dt)
#     elif option_type == "up-and-out call":
#         matval, veti, M, N, vetS, r =UOCall(S0,K,r,T,barrier,dS,dt)
#     elif option_type == "down-and-out call":
#         matval, veti, M, N, vetS, r =DOCall(S0,K,r,T,barrier,dS,dt)
#     elif option_type == "down-and-in call":
#         doc, _, _ = crank_nicolson(S0, K, r, T, sigma, dS, dt, "down-and-out call", barrier)
#         if doc < 0:
#             doc = 0
#             price = crank_nicolsonV(S0, K, r, T, sigma, dS, dt, "Call") - doc
#         else:
#             price = crank_nicolsonV(S0, K, r, T, sigma, dS, dt, "Call") - doc
#         price = crank_nicolsonV(S0, K, r, T, sigma, dS, dt, "Call") - doc
#         return price,_,_
#     elif option_type == "down-and-in put":
#         dop, _, _ = crank_nicolson(S0, K, r, T, sigma, dS, dt, "down-and-out put", barrier)
#         price = crank_nicolsonV(S0, K, r, T, sigma, dS, dt, "Put") - dop
#         return price,_,_
#     elif option_type == "up-and-in put":
#         uop, _, _ = crank_nicolson(S0, K, r, T, sigma, dS, dt, "up-and-out put", barrier)
#         if uop < 0:
#             uop = 0
#             price = crank_nicolsonV(S0, K, r, T, sigma, dS, dt, "Put") - uop
#         else: 
#             price = crank_nicolsonV(S0, K, r, T, sigma, dS, dt, "Put") - uop
#         return price,_,_
#     elif option_type == "up-and-in call":
#         uop, _, _ = crank_nicolson(S0, K, r, T, sigma, dS, dt, "up-and-out call", barrier)
#         price = crank_nicolsonV(S0, K, r, T, sigma, dS, dt, "Call") - uop
#         return price,_,_
#     elif option_type not in ("down-and-out put", "up-and-out put", "up-and-out call", "down-and-out call"):
#         st.error("UnboundLocalError: The variable 'veti' has not been assigned a value. This is likely due to an unsupported option type. Please select a valid option type, such as 'down-and-out put', 'up-and-out put', 'up-and-out call' or 'down-and-out call'.")
#         st.stop()        
        
#     # Set up the coefficients matrix
#     alpha = 0.25 * dt * (sigma**2 * (veti**2) - r * veti)
#     beta = -0.5 * dt * (sigma**2 * (veti**2) + r)
#     gamma = 0.25 * dt * (sigma**2 * (veti**2) + r * veti)

#     # Construct tridiagonal matrices
#     M1 = -np.diag(alpha[2:M], -1) + np.diag(1 - beta[1:M]) - np.diag(gamma[1:M-1], 1)
#     M2 = np.diag(alpha[2:M], -1) + np.diag(1 + beta[1:M]) + np.diag(gamma[1:M-1], 1)

#     # LU decomposition for efficient solving
#     LU, piv = lu_factor(M1)

#     # Solve the sequence of linear systems
#     lostval = np.zeros(M2.shape[1])

#     for j in range(N-1, -1, -1):
#         if len(lostval) > 1:
#             lostval[0] = alpha[1] * (matval[0, j] + matval[0, j+1])
#             lostval[-1] = gamma[-1] * (matval[-1, j] + matval[-1, j+1])
#         else:
#             lostval = lostval[0] + lostval[-1]

#         rhs = M2 @ matval[1:M, j+1] + lostval
#         matval[1:M, j] = lu_solve((LU, piv), rhs) 
    
#     price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
#     price = price_interp(S0)
                
#     return price, vetS, matval[:, 0]


# def DOPut(S0,K,r,T,barrier,dS,dt):
#     Smax = 2*max(S0,K)*np.exp(r*T)
#     M = round((Smax-barrier)/dS)
#     dS = (Smax-barrier)/M
#     N = round(T/dt)
#     dt = T/N
#     matval = np.zeros((M+1,N+1))
#     vetS = np.linspace(barrier,Smax,M+1)
#     veti = vetS/dS
#     vetj = np.arange(0, N+1)
    
#     matval[:,-1] = np.maximum(K-vetS,0)
#     matval[0,:] = 0
#     matval[-1,:] = 0
    
#     return matval, veti, M, N, vetS, r

# def UOPut(S0,K,r,T,barrier,dS,dt):
#     Smax = 2*max(S0,K)*np.exp(r*T)
#     M = round((barrier)/dS)
#     dS = (barrier)/M
#     N = round(T/dt)
#     dt = T/N
#     matval = np.zeros((M+1,N+1))
#     vetS = np.linspace(0,barrier,M+1)
#     veti = vetS/dS
#     vetj = np.arange(0, N+1)

#     matval[:,-1] = np.maximum(K-vetS,0)
#     matval[0,:] = K * np.exp(-r * (T - np.linspace(0, T, N + 1)))
#     matval[-1,:] = 0
    
#     return matval, veti, M, N, vetS, r
    
# def UOCall(S0,K,r,T,barrier,dS,dt):
#     Smax = 2*max(S0,K)*np.exp(r*T)
#     M = round((barrier)/dS)
#     dS = (barrier)/M
#     N = round(T/dt)
#     dt = T/N
#     matval = np.zeros((M+1,N+1))
#     vetS = np.linspace(0,barrier,M+1)
#     veti = vetS/dS
#     vetj = np.arange(0, N+1)

#     matval[:,-1] = np.maximum(vetS-K,0)
#     matval[0,:] = 0
#     matval[-1,:] = 0
    
#     return matval, veti, M, N, vetS, r

# def DOCall(S0,K,r,T,barrier,dS,dt):
#     Smax = 2*max(S0,K)*np.exp(r*T)
#     M = round((Smax-barrier)/dS)
#     dS = (Smax-barrier)/M
#     N = round(T/dt)
#     dt = T/N
#     matval = np.zeros((M+1,N+1))
#     vetS = np.linspace(barrier,Smax,M+1)
#     veti = vetS/dS
#     vetj = np.arange(0, N+1)
    
#     matval[:,-1] = np.maximum(vetS-K,0)
#     matval[0,:] = 0 
#     matval[-1,:] = Smax-K*np.exp(-r * dt*(N-vetj))
    
#     return matval, veti, M, N, vetS, r

# # Streamlit interface
# # st.title("Comparison of different numerical schemes and the analytical solution for barrier options")

# #  #S_max = st.sidebar.number_input("Maximum Stock Price (S_max)", value=200.0, step=1.0)
# # S0 = st.sidebar.number_input("Spot Price (S0)", value=100.0, step=1.0)
# # K = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0)
# # T = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0, step=0.1)
# # r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05, step=0.01)
# # sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01)
# # dS = st.sidebar.number_input("Stock Price Step (dS)", value=10.0, step=0.1)
# # dt = st.sidebar.number_input("Time Step (dt)", value=0.001, step=0.001)
# # option_type = st.sidebar.selectbox(
# #     "Option Type",
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
# # barrier = st.sidebar.number_input("Barrier", value=80.0, step=0.01)
# # q = st.sidebar.number_input("Dividend Yield (q)", value=0.03)
# # rebate = st.sidebar.number_input("Rebate", value=0.0)
# # numerical_method = st.sidebar.selectbox("Numerical method", ("Forward Euler", "Backward Euler", "Crank-Nicolson"))
     



# # if numerical_method == "Forward Euler":
# #     # Compute Forward Euler results
# #     num_price, S_grid, forward_euler_prices = forward_euler(S0, K, T, r, sigma, dS, dt, option_type, barrier)
# #     if num_price < 0:
# #         num_price = 0.0

# #     # Compute the analytical barrier option price at S0.
# #     analytical_price = barrier_option_price(S0, K, T, r, q, sigma, barrier, option_type)
    
# #     # Create an interpolation function from the numerical scheme results.
# #     price_interp = interp1d(S_grid, forward_euler_prices, kind='linear', fill_value="extrapolate")
# #     # Evaluate the interpolation on the same grid (or you could use a new grid if desired)
# #     forward_euler_prices_1d = price_interp(S_grid)
                                  
# #     # Create a DataFrame for the comparison at S0.
# #     df = pd.DataFrame({
# #         "Forward Euler Price": [np.round(num_price, 4)],
# #         "Analytical Price": [np.round(analytical_price, 4)],
# #         "Absolute Error": [np.round(np.abs(num_price - analytical_price), 4)]
# #     })
# #     st.subheader("Option Price Comparison at Spot Price (S0)")
# #     st.table(df)
    
# #     # Create a full grid for analytical prices over a broader range.
# #     S_grid_full = np.linspace(S_grid[0], S_grid[-1], 100)
# #     analytical_prices = np.array([barrier_option_price(S, K, T, r, q, sigma, barrier, option_type)
# #                                   for S in S_grid_full])
    
# #     # Plot the numerical (interpolated) and analytical prices.
# #     st.subheader("Comparison of Prices Across All Stock Prices")
# #     fig = go.Figure()
# #     fig.add_trace(go.Scatter(
# #         x=S_grid,
# #         y=forward_euler_prices_1d,
# #         mode="markers",
# #         name="Forward Euler Prices",
# #         marker=dict(color="red", size=6)
# #     ))
# #     fig.add_trace(go.Scatter(
# #         x=S_grid_full,
# #         y=analytical_prices,
# #         mode="lines",
# #         name="Analytical Barrier Prices",
# #         line=dict(color="blue", width=2)
# #     ))
# #     fig.update_layout(
# #         title="Option Prices: Forward Euler vs Analytical",
# #         xaxis_title="Stock Price (S)",
# #         yaxis_title="Option Price (V)",
# #         legend_title="Method",
# #         width=800,
# #         height=500
# #     )
# #     st.plotly_chart(fig)

    
# # elif numerical_method == "Backward Euler":
# #     # Compute Forward Euler results
# #     price, S_grid, backward_euler_prices = backward_euler(S0, K, r, T, sigma, dS, dt, option_type, barrier)
    
# #     if price < 0:
# #         price = 0.0
# #     else:
# #         pass

# #     # Compute analytical Black-Scholes prices
# #     analytical_price = barrier_option_price(S0, K, T, r, q, sigma, barrier, option_type)
# #     # Compute analytical prices over the entire grid S_grid.
# #     analytical_prices =barrier_option_price(S_grid, K, T, r, q, sigma, barrier, option_type)
                                  

# #     # Compute the absolute errors.
# #     absolute_error = np.abs(np.array(backward_euler_prices) - analytical_prices)


# #     # Create a DataFrame for comparison at S0
# #     df = pd.DataFrame({
# #         "Backward Euler Price": [np.round(price, 4)],
# #         "Analytical Price": [analytical_price],
# #         "Absolute Error": [np.abs(price - analytical_price)],
# #     })

# #     # Display the table for the spot price
# #     st.subheader("Option Price Comparison at Spot Price (S0)")
# #     st.table(df)

# #     # Plot the results
# #     st.subheader("Comparison of Prices Across All Stock Prices")
# #     fig = go.Figure()

# #     # Scatter plot for Forward Euler
# #     fig.add_trace(go.Scatter(
# #         x=S_grid, 
# #         y=backward_euler_prices, 
# #         mode="markers", 
# #         name="Backward Euler Prices",
# #         marker=dict(color="red", size=6)
# #     ))

# #     # Line plot for Analytical Black-Scholes
# #     fig.add_trace(go.Scatter(
# #         x=S_grid, 
# #         y=analytical_prices, 
# #         mode="lines", 
# #         name="Analytical Black-Scholes Prices",
# #         line=dict(color="blue", width=2)
# #     ))

# #     fig.update_layout(
# #         title="Option Prices: Backward Euler vs Analytical",
# #         xaxis_title="Stock Price (S)",
# #         yaxis_title="Option Price (V)",
# #         legend_title="Method",
# #         width=800,
# #         height=500
# #     )
# #     st.plotly_chart(fig)


# # elif numerical_method == "Crank-Nicolson":
# #     # Compute Forward Euler results
# #     price, S_grid, crank_nicolson_prices= crank_nicolson(S0, K, r, T, sigma, dS, dt, option_type, barrier)
    
# #     if price < 0:
# #         price = 0.0
# #     else:
# #         pass

# #     # Compute analytical Black-Scholes prices
# #     #analytical_price = black_scholes(S0, K, T, r, sigma, option_type)
# #     #analytical_prices = black_scholes(S_grid, K, T, r, sigma, option_type)
# #     # Find the index closest to S0
# #     index_S0 = (np.abs(S_grid - S0)).argmin()


# #     # Create a DataFrame for comparison at S0
# #     df = pd.DataFrame({
# #         "Crank Nicolson Price": [np.round(price, 4)],
# #         #"Analytical Price": [analytical_price],
# #         #"Absolute Error": [np.abs(price - analytical_price)],
# #     })

# #     # Display the table for the spot price
# #     st.subheader("Option Price Comparison at Spot Price (S0)")
# #     st.table(df)

# #     # Plot the results
# #     st.subheader("Comparison of Prices Across All Stock Prices")
# #     fig = go.Figure()

# #     # Scatter plot for Forward Euler
# #     fig.add_trace(go.Scatter(
# #         x=S_grid, 
# #         y=crank_nicolson_prices, 
# #         mode="markers", 
# #         name="Crank Nicolson Prices",
# #         marker=dict(color="red", size=6)
# #     ))

# #     #Line plot for Analytical Black-Scholes
# #     # fig.add_trace(go.Scatter(
# #     #     x=S_grid, 
# #     #     y=analytical_prices, 
# #     #     mode="lines", 
# #     #     name="Analytical Black-Scholes Prices",
# #     #     line=dict(color="blue", width=2)
# #     # ))

# #     fig.update_layout(
# #         title="Option Prices: Crank Nicolson vs Analytical",
# #         xaxis_title="Stock Price (S)",
# #         yaxis_title="Option Price (V)",
# #         legend_title="Method",
# #         width=800,
# #         height=500
# #     )
# #     st.plotly_chart(fig)




# st.title("Comparison of different numerical schemes and the analytical solution for barrier options")

# S0 = st.sidebar.number_input("Spot Price (S0)", value=100.0, step=1.0)
# K = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0)
# T = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0, step=0.1)
# r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05, step=0.01)
# sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01)
# dS = st.sidebar.number_input("Stock Price Step (dS)", value=10.0, step=0.1)
# dt = st.sidebar.number_input("Time Step (dt)", value=0.001, step=0.001)
# option_type = st.sidebar.selectbox(
#     "Option Type",
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
# barrier = st.sidebar.number_input("Barrier", value=80.0, step=0.01)
# q = st.sidebar.number_input("Dividend Yield (q)", value=0.03)
# rebate = st.sidebar.number_input("Rebate", value=0.0)
# numerical_method = st.sidebar.selectbox("Numerical method", ("Forward Euler", "Backward Euler", "Crank-Nicolson"))

# st.write(f"Selected: {numerical_method} for {option_type}, Barrier={barrier}")

# ################################################################################
# # 1) Compute PDE solution at S0, also keep the PDE “slice” at t=0 for all S
# ################################################################################

# if numerical_method == "Forward Euler":
#     num_price, S_grid, pde_values_t0 = forward_euler(
#         S0, K, T, r, sigma, dS, dt, option_type, barrier
#     )
#     # pde_values_t0 is the array of PDE values at time=0 over S_grid
#     # If negative, clamp to 0
#     if num_price < 0:
#         num_price = 0.0

#     # Analytical price at S0:
#     analytical_price_S0 = barrier_option_price(S0, K, T, r, q, sigma, barrier, option_type)

#     # Now build a *common* x-axis for plotting. We can simply use S_grid itself,
#     # or a denser grid. Let’s do S_grid for direct comparison.
#     # Evaluate PDE solution is pde_values_t0
#     # Evaluate analytical solution at each S in S_grid:
#     analytic_values = [
#         barrier_option_price(s, K, T, r, q, sigma, barrier, option_type)
#         for s in S_grid
#     ]

#     # Build DataFrame showing the single point (S0) comparison
#     df = pd.DataFrame({
#         "Scheme Price at S0": [np.round(num_price,4)],
#         "Analytical Price at S0": [np.round(analytical_price_S0,4)],
#         "Abs Error": [round(abs(num_price - analytical_price_S0),4)]
#     })
#     st.subheader("Option Price Comparison at Spot Price (S0)")
#     st.table(df)

#     # Plot
#     st.subheader("Comparison of Prices Across Stock-Price Grid")
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=S_grid,
#         y=pde_values_t0,
#         mode="markers+lines",
#         name="Forward Euler PDE",
#         marker=dict(color="red", size=5)
#     ))
#     fig.add_trace(go.Scatter(
#         x=S_grid,
#         y=analytic_values,
#         mode="lines",
#         name="Analytical Barrier",
#         line=dict(color="blue", width=2)
#     ))
#     fig.update_layout(
#         title="Forward Euler vs Analytical",
#         xaxis_title="Stock Price (S)",
#         yaxis_title="Option Value",
#         legend_title="Method"
#     )
#     st.plotly_chart(fig)


# elif numerical_method == "Backward Euler":
#     price_BE, S_grid, pde_values_t0 = backward_euler(
#         S0, K, r, T, sigma, dS, dt, option_type, barrier
#     )
#     if price_BE < 0:
#         price_BE = 0.0

#     # Analytical price at S0
#     analytical_price_S0 = barrier_option_price(S0, K, T, r, q, sigma, barrier, option_type)
#     # Analytical curve
#     analytic_values = [
#         barrier_option_price(s, K, T, r, q, sigma, barrier, option_type)
#         for s in S_grid
#     ]

#     df = pd.DataFrame({
#         "Scheme Price at S0": [np.round(price_BE,4)],
#         "Analytical Price at S0": [np.round(analytical_price_S0,4)],
#         "Abs Error": [round(abs(price_BE - analytical_price_S0),4)]
#     })
#     st.subheader("Option Price Comparison at Spot Price (S0)")
#     st.table(df)

#     # Plot
#     st.subheader("Comparison of Prices Across Stock-Price Grid")
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=S_grid,
#         y=pde_values_t0,
#         mode="markers+lines",
#         name="Backward Euler PDE",
#         marker=dict(color="red", size=5)
#     ))
#     fig.add_trace(go.Scatter(
#         x=S_grid,
#         y=analytic_values,
#         mode="lines",
#         name="Analytical Barrier",
#         line=dict(color="blue", width=2)
#     ))
#     fig.update_layout(
#         title="Backward Euler vs Analytical",
#         xaxis_title="Stock Price (S)",
#         yaxis_title="Option Value",
#         legend_title="Method"
#     )
#     st.plotly_chart(fig)


# elif numerical_method == "Crank-Nicolson":
#     price_CN, S_grid, pde_values_t0 = crank_nicolson(
#         S0, K, r, T, sigma, dS, dt, option_type, barrier
#     )
#     if price_CN < 0:
#         price_CN = 0.0

#     # Analytical price at S0
#     analytical_price_S0 = barrier_option_price(S0, K, T, r, q, sigma, barrier, option_type)
#     # Analytical curve
#     analytic_values = [
#         barrier_option_price(s, K, T, r, q, sigma, barrier, option_type)
#         for s in S_grid
#     ]

#     df = pd.DataFrame({
#         "Scheme Price at S0": [np.round(price_CN,4)],
#         "Analytical Price at S0": [np.round(analytical_price_S0,4)],
#         "Abs Error": [round(abs(price_CN - analytical_price_S0),4)]
#     })
#     st.subheader("Option Price Comparison at Spot Price (S0)")
#     st.table(df)

#     # Plot
#     st.subheader("Comparison of Prices Across Stock-Price Grid")
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=S_grid,
#         y=pde_values_t0,
#         mode="markers+lines",
#         name="Crank-Nicolson PDE",
#         marker=dict(color="red", size=5)
#     ))
#     fig.add_trace(go.Scatter(
#         x=S_grid,
#         y=analytic_values,
#         mode="lines",
#         name="Analytical Barrier",
#         line=dict(color="blue", width=2)
#     ))
#     fig.update_layout(
#         title="Crank-Nicolson vs Analytical",
#         xaxis_title="Stock Price (S)",
#         yaxis_title="Option Value",
#         legend_title="Method"
#     )
#     st.plotly_chart(fig)
####################################################################################################################################################################
#MAIN
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.linalg import lu_factor, lu_solve

def calc_d1(S0, K, r, q, sigma, T):
    return (np.log(S0 / K) + (r - q + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))

def calc_d2(S0, K, r, q, sigma, T):
    return calc_d1(S0, K, r, q, sigma, T) - sigma * np.sqrt(T)

def calc_c(S0, K, r, q, sigma, T):
    d1 = calc_d1(S0, K, r, q, sigma, T)
    d2 = calc_d2(S0, K, r, q, sigma, T)
    return (S0 * np.exp(-q*T)*norm.cdf(d1)
            - K * np.exp(-r*T)*norm.cdf(d2))

def calc_p(S0, K, r, q, sigma, T):
    d1 = calc_d1(S0, K, r, q, sigma, T)
    d2 = calc_d2(S0, K, r, q, sigma, T)
    return (K * np.exp(-r*T)*norm.cdf(-d2)
            - S0 * np.exp(-q*T)*norm.cdf(-d1))

def calc_lambda(r, q, sigma):
    # λ = (r - q + σ²/2) / σ²
    return (r - q + 0.5 * sigma**2) / (sigma**2)

def calc_y(barrier, S0, K, T, sigma, r, q):
    """
    y = [ln(barrier^2/(S0*K)) / (sigma*sqrt(T))] + λ * sigma * sqrt(T)
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
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "Call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return price
    elif option_type == "Put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return price


# ------------------------------
# 2) Main barrier pricing function
# ------------------------------

def barrier_option_price(S0, K, T, r, q, sigma, barrier, option_type):
    """
    Returns the price of a barrier option (various knock-in/out types).
    Matches standard formulas from texts like Hull, with care to keep
    exponents and sign conventions correct.
    """
    x1 = calc_x1(S0, barrier, T, sigma, r, q)
    y1 = calc_y1(S0, barrier, T, sigma, r, q)
    c = calc_c(S0, K, r, q, sigma, T)
    p = calc_p(S0, K, r, q, sigma, T)
    lam = calc_lambda(r, q, sigma)
    y  = calc_y(barrier, S0, K, T, sigma, r, q)

    # --------------------------------
    # Down-and-in Call
    # --------------------------------
    
    if option_type == 'down-and-in call' and barrier <= K and S0 <= barrier:
        vanilla = black_scholes(S0, K, T, r, sigma, "Call")
        return vanilla
    
    elif option_type == 'down-and-in call' and barrier <= K:
        # cdi, for barrier <= K
        cdi = (S0 * np.exp(-q*T) * (barrier/S0)**(2*lam) * norm.cdf(y)
               - K * np.exp(-r*T) * (barrier/S0)**(2*lam - 2)
                 * norm.cdf(y - sigma*np.sqrt(T)))
        return cdi

    elif option_type == 'down-and-in call' and barrier >= K:
        # cdi = c - cdo. So we compute cdo from the standard expression
        # cdo = ...
        # Then cdi = c - cdo
        term1 = S0*np.exp(-q*T)*norm.cdf(x1)
        term2 = K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
        term3 = S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(y1)
        term4 = K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)*norm.cdf(y1 - sigma*np.sqrt(T))
        cdo   = term1 - term2 - term3 + term4
        if cdo < 0:
            cdo = 0
            cdi   = c - cdo
            return cdi
        else:
            cdi = c - cdo
            return cdi

    # --------------------------------
    # Down-and-out Call
    # --------------------------------
    elif option_type == 'down-and-out call' and barrier <= K:

        cdi = (S0 * np.exp(-q*T) * (barrier/S0)**(2*lam) * norm.cdf(y)
            - K * np.exp(-r*T) * (barrier/S0)**(2*lam - 2)
                * norm.cdf(y - sigma*np.sqrt(T)))
        cdo = c - cdi
        if cdo > 0:
            return cdo
        else:
            return 0

    elif option_type == 'down-and-out call' and barrier >= K:
        # This is the “If barrier > K” formula for the down-and-out call
        term1 = S0 * np.exp(-q*T)*norm.cdf(x1)
        term2 = K  * np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
        term3 = S0 * np.exp(-q*T)*((barrier/S0)**(2*lam))*norm.cdf(y1)
        term4 = K  * np.exp(-r*T)*((barrier/S0)**(2*lam - 2))*norm.cdf(y1 - sigma*np.sqrt(T))
        cdo   = term1 - term2 - term3 + term4
        
        if cdo < 0:
            return 0
        else:
            return cdo

    # --------------------------------
    # Up-and-in Call
    # --------------------------------
    elif option_type == 'up-and-in call' and barrier > K:
        # Standard up-and-in call for barrier > K
        cui = (S0*np.exp(-q*T)*norm.cdf(x1)
               - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
               - S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*(norm.cdf(-y) - norm.cdf(-y1))
               + K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)
                 * (norm.cdf(-y + sigma*np.sqrt(T))
                    - norm.cdf(-y1 + sigma*np.sqrt(T))))
        return cui

    elif option_type == 'up-and-in call' and barrier <= K:
        # If barrier is below K, the up-and-in call is effectively the same as c
        # or 0, depending on your setup.  Typically if barrier < S0 < K,
        # the option knocks in only if S0 goes above barrier.  If you are sure
        # you want to treat it as simply c, do so here:
        return c

    # --------------------------------
    # Up-and-out Call
    # --------------------------------
    elif option_type == 'up-and-out call' and barrier <= K:
        # If the barrier barrier <= K is below the current spot,
        # often up-and-out call is worthless if it is truly "up" barrier?
        return 0.0

    elif option_type == 'up-and-out call' and barrier > K:
        cui = (S0*np.exp(-q*T)*norm.cdf(x1)
               - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
               - S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*(norm.cdf(-y) - norm.cdf(-y1))
               + K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)
                 * (norm.cdf(-y + sigma*np.sqrt(T))
                    - norm.cdf(-y1 + sigma*np.sqrt(T))))
        cuo = c - cui
        if cuo > 0:
            return cuo
        else:
            return 0
        

    # --------------------------------
    # Up-and-in Put
    # --------------------------------
    elif option_type == 'up-and-in put' and barrier >= K and barrier <= S0:
        pui = black_scholes(S0,K,T,r,sigma,"Put")
        return pui
    elif option_type == 'up-and-in put' and barrier >= K:
        pui = (-S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(-y)
               + K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)
                 * norm.cdf(-y + sigma*np.sqrt(T)))
        return pui
    
        # --------------------------------
    elif option_type == 'up-and-in put' and barrier <= K:
        puo = (
            -S0*np.exp(-q*T)*norm.cdf(-x1)
            + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
            + S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(-y1)
            - K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)*norm.cdf(-y1 + sigma*np.sqrt(T))
        )
        if puo < 0:
            puo = 0
            pui = black_scholes(S0,K,T,r,sigma,"Put")
            return pui
        else:
            pui = black_scholes(S0,K,T,r,sigma,"Put") - puo
        
        return pui
    
    elif option_type == 'up-and-in put' and barrier <= K:
        # up-and-in put is the difference p - up-and-out put
        # but for the simplified logic, we can just return p if the barrier is < K
        return p

    # --------------------------------
    # Up-and-out Put
    # --------------------------------
    elif option_type == 'up-and-out put' and barrier >= K:
        # puo = p - pui
        pui = (-S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(-y)
               + K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)
                 * norm.cdf(-y + sigma*np.sqrt(T)))
        if pui > 0:
            puo = p - pui
            return puo
        else:
            pui = 0
            puo = p - pui
            return puo

    elif option_type == 'up-and-out put' and barrier <= K:
        # Standard formula for barrier <= K
        puo = (
            -S0*np.exp(-q*T)*norm.cdf(-x1)
            + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
            + S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(-y1)
            - K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)*norm.cdf(-y1 + sigma*np.sqrt(T))
        )
        if puo < 0:
            puo = 0
            return puo
        else:
            return puo


    # --------------------------------
    # Down-and-in Put
    # --------------------------------
    elif option_type == 'down-and-in put' and barrier < K and S0 < barrier:
        vanilla = black_scholes(S0, K, T, r, sigma, "Put")
        return vanilla
    
    elif option_type == 'down-and-in put' and barrier > K:
        # If the barrier is above K, we often treat the down-and-in put as p
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

    # --------------------------------
    # Down-and-out Put
    # --------------------------------
    elif option_type == 'down-and-out put' and barrier > K:
        # Typically worthless if barrier > K in certain setups
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
        if pdo > 0:
            return pdo
        else:
            return 0

    # Fallback
    return None


# ################################################################################
# # 2) PDE for a Vanilla Call on [0, S_max]
# ################################################################################

###############################################################################
# 1) Vanilla PDEs (Forward Euler)
###############################################################################
def forward_euler_vanilla_call(S0, K, T, r, sigma, dS, dt):
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


def forward_euler_vanilla_put(S0, K, T, r, sigma, dS, dt):
    """
    Forward Euler PDE for a vanilla European put on [0, S_max].
    Returns: (priceVan, S_grid, V0).
    """
    S_max = 2 * max(S0, K) * np.exp(r * T)
    M = int(S_max / dS)
    N = int(T / dt)
    dS = S_max / M
    dt = T / N

    S_grid = np.linspace(0, S_max, M + 1)
    V = np.zeros((N + 1, M + 1))

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

    # Forward Euler stepping
    for n in range(N, 0, -1):
        for j in range(1, M):
            V[n - 1, j] = a[j] * V[n, j - 1] + b[j] * V[n, j] + c[j] * V[n, j + 1]

    # Interpolate
    interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
    priceVan = float(interp_fn(S0))
    return priceVan, S_grid, V[0, :]


###############################################################################
# 2) Barrier PDEs (Forward Euler)
###############################################################################
def forward_euler_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type):
    """
    Forward Euler for a knock-out call (either 'down-and-out' or 'up-and-out').
    barrier_type = 'down' or 'up'
    
    We zero out the region beyond the barrier:
      - If 'down-and-out', zero for S <= barrier
      - If 'up-and-out',   zero for S >= barrier
    """
    S_max = 2 * max(S0, K) * np.exp(r * T)
    M = int(S_max / dS)
    N = int(T / dt)
    dS = S_max / M
    dt = T / N

    S_grid = np.linspace(0, S_max, M + 1)
    V = np.zeros((N + 1, M + 1))

    # Terminal payoff for a call
    payoff = np.maximum(S_grid - K, 0.0)
    if barrier_type == 'down':
        # down-and-out => zero payoff for S <= barrier
        payoff[S_grid <= barrier] = 0.0
    else:
        # up-and-out => zero payoff for S >= barrier
        payoff[S_grid >= barrier] = 0.0
    V[-1, :] = payoff

    # Boundary conditions for a call
    t_arr = np.linspace(0, T, N + 1)
    for i in range(N + 1):
        tau = T - t_arr[i]
        V[i, 0]   = 0.0
        V[i, -1]  = S_max - K * np.exp(-r * tau)

    # PDE coefficients
    j_arr = np.arange(M + 1)
    a = 0.5 * dt * (sigma**2 * j_arr**2 - r * j_arr)
    b = 1.0 - dt * (sigma**2 * j_arr**2 + r)
    c = 0.5 * dt * (sigma**2 * j_arr**2 + r * j_arr)

    # Forward Euler stepping
    for n in range(N, 0, -1):
        for j in range(1, M):
            V[n - 1, j] = a[j] * V[n, j - 1] + b[j] * V[n, j] + c[j] * V[n, j + 1]

        # Knock-out region
        if barrier_type == 'down':
            # zero out for S <= barrier
            V[n - 1, S_grid <= barrier] = 0.0
        else:
            # zero out for S >= barrier
            V[n - 1, S_grid >= barrier] = 0.0

    # Price at S0
    interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
    priceKO = float(interp_fn(S0))
    return priceKO, S_grid, V[0, :]


def forward_euler_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type):
    """
    Forward Euler for a knock-out put (either 'down-and-out' or 'up-and-out').
    barrier_type = 'down' or 'up'
    """
    S_max = 2 * max(S0, K) * np.exp(r * T)
    M = int(S_max / dS)
    N = int(T / dt)
    dS = S_max / M
    dt = T / N

    S_grid = np.linspace(0, S_max, M + 1)
    V = np.zeros((N + 1, M + 1))

    # Terminal payoff for a put
    payoff = np.maximum(K - S_grid, 0.0)
    if barrier_type == 'down':
        # down-and-out => zero payoff for S <= barrier
        payoff[S_grid <= barrier] = 0.0
    else:
        # up-and-out => zero payoff for S >= barrier
        payoff[S_grid >= barrier] = 0.0
    V[-1, :] = payoff

    # Boundary conditions for a put
    t_arr = np.linspace(0, T, N + 1)
    for i in range(N + 1):
        tau = T - t_arr[i]
        # For a put: V(0,t) ~ K e^{-r tau},  V(Smax,t) ~ 0
        V[i, 0]   = K * np.exp(-r * tau)
        V[i, -1]  = 0.0

    # PDE coefficients
    j_arr = np.arange(M + 1)
    a = 0.5 * dt * (sigma**2 * j_arr**2 - r * j_arr)
    b = 1.0 - dt * (sigma**2 * j_arr**2 + r)
    c = 0.5 * dt * (sigma**2 * j_arr**2 + r * j_arr)

    # Forward Euler stepping
    for n in range(N, 0, -1):
        for j in range(1, M):
            V[n - 1, j] = a[j] * V[n, j - 1] + b[j] * V[n, j] + c[j] * V[n, j + 1]

        # Knock-out region
        if barrier_type == 'down':
            V[n - 1, S_grid <= barrier] = 0.0
        else:
            V[n - 1, S_grid >= barrier] = 0.0

    # Price at S0
    interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
    priceKO = float(interp_fn(S0))
    return priceKO, S_grid, V[0, :]


###############################################################################
# 3) Main wrapper: forward_euler(...)
###############################################################################
def forward_euler(S0, K, T, r, sigma, dS, dt, barrier, option_type):
    """
    Main wrapper for forward Euler barrier options.
    We handle the 'knock-out' case directly by zeroing out the barrier region.
    We handle the 'knock-in' case via:
         knock_in = vanilla - knock_out
    """
    # ---------------------------
    # A) DOWN-AND-OUT CALL
    # ---------------------------
    if option_type == "down-and-out call":
        return forward_euler_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down')

    # ---------------------------
    # B) DOWN-AND-IN CALL
    #    = vanilla call - down-and-out call
    # ---------------------------
    elif option_type == "down-and-in call":
        # 1) Price of down-and-out call
        priceDOC, Sg_DO, PDE_DO = forward_euler_knock_out_call(
            S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down'
        )
        # 2) Price of vanilla call
        priceVan, Sg_van, PDE_van = forward_euler_vanilla_call(S0, K, T, r, sigma, dS, dt)
        # 3) In-out parity
        priceDin = priceVan - priceDOC
        PDE_din  = PDE_van - PDE_DO
        return priceDin, Sg_van, PDE_din

    # ---------------------------
    # C) DOWN-AND-OUT PUT
    # ---------------------------
    elif option_type == "down-and-out put":
        return forward_euler_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down')

    # ---------------------------
    # D) DOWN-AND-IN PUT
    #    = vanilla put - down-and-out put
    # ---------------------------
    elif option_type == "down-and-in put":
        # 1) Price of down-and-out put
        priceDOP, Sg_DO, PDE_DO = forward_euler_knock_out_put(
            S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down'
        )
        # 2) Price of vanilla put
        priceVan, Sg_van, PDE_van = forward_euler_vanilla_put(S0, K, T, r, sigma, dS, dt)
        # 3) In-out parity
        priceDin = priceVan - priceDOP
        PDE_din  = PDE_van - PDE_DO
        return priceDin, Sg_van, PDE_din

    # ---------------------------
    # E) UP-AND-OUT CALL
    # ---------------------------
    elif option_type == "up-and-out call":
        return forward_euler_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up')

    # ---------------------------
    # F) UP-AND-IN CALL
    #    = vanilla call - up-and-out call
    # ---------------------------
    elif option_type == "up-and-in call":
        priceUOC, Sg_UO, PDE_UO = forward_euler_knock_out_call(
            S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up'
        )
        priceVan, Sg_van, PDE_van = forward_euler_vanilla_call(S0, K, T, r, sigma, dS, dt)
        priceUIC = priceVan - priceUOC
        PDE_uic  = PDE_van - PDE_UO
        return priceUIC, Sg_van, PDE_uic

    # ---------------------------
    # G) UP-AND-OUT PUT
    # ---------------------------
    elif option_type == "up-and-out put":
        return forward_euler_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up')

    # ---------------------------
    # H) UP-AND-IN PUT
    #    = vanilla put - up-and-out put
    # ---------------------------
    elif option_type == "up-and-in put":
        priceUOP, Sg_UO, PDE_UO = forward_euler_knock_out_put(
            S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up'
        )
        priceVan, Sg_van, PDE_van = forward_euler_vanilla_put(S0, K, T, r, sigma, dS, dt)
        priceUIP = priceVan - priceUOP
        PDE_uip  = PDE_van - PDE_UO
        Sg = Sg_van-Sg_UO
        return priceUIP, Sg_van, PDE_uip

    return None
# def forward_euler_vanilla_call(S0, K, T, r, sigma, dS, dt):
#     """
#     Forward Euler PDE for a vanilla European call on [0, S_max].
#     Returns: (priceVan, S_grid, V0) => scalar price at S0, S-array, PDE values at time 0.
#     """
#     S_max = 2 * max(S0, K) * np.exp(r*T)
#     M = int(S_max / dS)
#     N = int(T / dt)
#     dS = S_max / M
#     dt = T / N

#     S_grid = np.linspace(0, S_max, M + 1)
#     V = np.zeros((N + 1, M + 1))

#     # Terminal payoff
#     V[-1, :] = np.maximum(S_grid - K, 0.0)

#     # Boundary conditions
#     t_arr = np.linspace(0, T, N+1)
#     for i in range(N+1):
#         tau = T - t_arr[i]
#         V[i, 0] = 0.0
#         V[i, -1] = S_max - K*np.exp(-r*tau)

#     # PDE Coeffs
#     j_arr = np.arange(M+1)
#     a = 0.5*dt*(sigma**2 * j_arr**2 - r*j_arr)
#     b = 1.0 - dt*(sigma**2*j_arr**2 + r)
#     c = 0.5*dt*(sigma**2 * j_arr**2 + r*j_arr)

#     # Forward Euler stepping
#     for n in range(N, 0, -1):
#         for j in range(1, M):
#             V[n-1, j] = a[j]*V[n, j-1] + b[j]*V[n, j] + c[j]*V[n, j+1]

#     # Interpolate to get price at S0
#     interp_fn = interp1d(S_grid, V[0,:], kind='linear', fill_value='extrapolate')
#     priceVan = float(interp_fn(S0))
#     return priceVan, S_grid, V[0,:]

# def forward_euler_vanilla_put(S0, K, T, r, sigma, dS, dt):
#     """
#     Forward Euler PDE for a vanilla European call on [0, S_max].
#     Returns: (priceVan, S_grid, V0) => scalar price at S0, S-array, PDE values at time 0.
#     """
#     S_max = 2 * max(S0, K) * np.exp(r*T)
#     M = int(S_max / dS)
#     N = int(T / dt)
#     dS = S_max / M
#     dt = T / N

#     S_grid = np.linspace(0, S_max, M + 1)
#     V = np.zeros((N + 1, M + 1))

#     # Terminal payoff
#     V[-1, :] = np.maximum(K - S_grid, 0.0)

#     # Boundary conditions
#     t_arr = np.linspace(0, T, N+1)
#     for i in range(N+1):
#         tau = T - t_arr[i]
#         V[i, 0] = 0
#         V[i, -1] = K*np.exp(-r*tau)
#         #V[i,-1] = 0

#     # PDE Coeffs
#     j_arr = np.arange(M+1)
#     a = 0.5*dt*(sigma**2 * j_arr**2 - r*j_arr)
#     b = 1.0 - dt*(sigma**2*j_arr**2 + r)
#     c = 0.5*dt*(sigma**2 * j_arr**2 + r*j_arr)

#     # Forward Euler stepping
#     for n in range(N, 0, -1):
#         for j in range(1, M):
#             V[n-1, j] = a[j]*V[n, j-1] + b[j]*V[n, j] + c[j]*V[n, j+1]

#     # Interpolate to get price at S0
#     interp_fn = interp1d(S_grid, V[0,:], kind='linear', fill_value='extrapolate')
#     priceVan = float(interp_fn(S0))
#     return priceVan, S_grid, V[0,:]




# def forward_euler(S0, K, T, r, sigma, dS, dt, barrier, option_type):
#     if option_type == "down-and-out call":
#         """
#         Solve PDE for a down-and-out call on [0, S_max].
#         We'll zero out the payoff & values for S <= barrier.
#         """
#         S_max = 2 * max(S0, K) * np.exp(r*T)
#         M = int(S_max / dS)
#         N = int(T / dt)
#         dS = S_max / M
#         dt = T / N

#         S_grid = np.linspace(0, S_max, M+1)
#         V = np.zeros((N+1, M+1))

#         # Terminal payoff: call for S>barrier, else 0
#         payoff = np.maximum(S_grid - K, 0.0)
#         payoff[S_grid <= barrier] = 0.0
#         V[-1,:] = payoff

#         # Boundaries
#         t_arr = np.linspace(0, T, N+1)
#         for i in range(N+1):
#             tau = T - t_arr[i]
#             V[i, 0]   = 0.0
#             V[i, -1]  = S_max - K*np.exp(-r*tau)

#         # PDE Coeffs
#         j_arr = np.arange(M+1)
#         a = 0.5*dt*(sigma**2 * j_arr**2 - r*j_arr)
#         b = 1.0 - dt*(sigma**2*j_arr**2 + r)
#         c = 0.5*dt*(sigma**2 * j_arr**2 + r*j_arr)

#         # Forward Euler stepping
#         for n in range(N, 0, -1):
#             for j in range(1, M):
#                 V[n-1, j] = a[j]*V[n, j-1] + b[j]*V[n, j] + c[j]*V[n, j+1]
#             # zero out for S <= barrier
#             for j in range(M+1):
#                 if S_grid[j] <= barrier:
#                     V[n-1, j] = 0.0

#         # Interpolate to get price at S0
#         interp_fn = interp1d(S_grid, V[0,:], kind='linear', fill_value='extrapolate')
#         priceDoc = float(interp_fn(S0))
#         return priceDoc, S_grid, V[0,:]
    
#     elif option_type == "down-and-in call":
#         S_max = 2 * max(S0, K) * np.exp(r*T)
#         M = int(S_max / dS)
#         N = int(T / dt)
#         dS = S_max / M
#         dt = T / N

#         S_grid = np.linspace(0, S_max, M+1)
#         V = np.zeros((N+1, M+1))

#         # Terminal payoff: call for S>barrier, else 0
#         payoff = np.maximum(S_grid - K, 0.0)
#         payoff[S_grid <= barrier] = 0.0
#         V[-1,:] = payoff

#         # Boundaries
#         t_arr = np.linspace(0, T, N+1)
#         for i in range(N+1):
#             tau = T - t_arr[i]
#             V[i, 0]   = 0.0
#             V[i, -1]  = S_max - K*np.exp(-r*tau)

#         # PDE Coeffs
#         j_arr = np.arange(M+1)
#         a = 0.5*dt*(sigma**2 * j_arr**2 - r*j_arr)
#         b = 1.0 - dt*(sigma**2*j_arr**2 + r)
#         c = 0.5*dt*(sigma**2 * j_arr**2 + r*j_arr)

#         # Forward Euler stepping
#         for n in range(N, 0, -1):
#             for j in range(1, M):
#                 V[n-1, j] = a[j]*V[n, j-1] + b[j]*V[n, j] + c[j]*V[n, j+1]
#             # zero out for S <= barrier
#             for j in range(M+1):
#                 if S_grid[j] <= barrier:
#                     V[n-1, j] = 0.0

#         # Interpolate to get price at S0
#         interp_fn = interp1d(S_grid, V[0,:], kind='linear', fill_value='extrapolate')
#         priceDoc = float(interp_fn(S0))
        
#         priceVan, S_gridV, PDE_van = forward_euler_vanilla_call(S0, K, T, r, sigma, dS, dt)
        
#         PDE_din = PDE_van - V[0,:]
#         priceDin = priceVan - priceDoc
        
#         return priceDin, S_gridV, PDE_din
    
#     elif option_type == "down-and-out put":
#         S_max = 2 * max(S0, K) * np.exp(r*T)
#         M = int(S_max / dS)
#         N = int(T / dt)
#         dS = S_max / M
#         dt = T / N

#         S_grid = np.linspace(0, S_max, M+1)
#         V = np.zeros((N+1, M+1))

#         # Terminal payoff: call for S>barrier, else 0
#         payoff = np.maximum(K - S_grid, 0.0)
#         payoff[S_grid <= barrier] = 0.0
#         V[-1,:] = payoff

#         # Boundaries
#         t_arr = np.linspace(0, T, N+1)
#         for i in range(N+1):
#             tau = T - t_arr[i]
#             V[i, 0]   = K * np.exp(-r*tau)
#             V[i, -1]  = 0.0
#             #V[i, -1]  = K * np.exp(-r*tau)

#         # PDE Coeffs
#         j_arr = np.arange(M+1)
#         a = 0.5*dt*(sigma**2 * j_arr**2 - r*j_arr)
#         b = 1.0 - dt*(sigma**2*j_arr**2 + r)
#         c = 0.5*dt*(sigma**2 * j_arr**2 + r*j_arr)

#         # Forward Euler stepping
#         for n in range(N, 0, -1):
#             for j in range(1, M):
#                 V[n-1, j] = a[j]*V[n, j-1] + b[j]*V[n, j] + c[j]*V[n, j+1]
#             # zero out for S <= barrier
#             for j in range(M+1):
#                 if S_grid[j] <= barrier:
#                     V[n-1, j] = 0.0

#         # Interpolate to get price at S0
#         interp_fn = interp1d(S_grid, V[0,:], kind='linear', fill_value='extrapolate')
#         priceDop = float(interp_fn(S0))
#         return priceDop, S_grid, V[0,:]
    
#     elif option_type == "down-and-in put":
#         S_max = 2 * max(S0, K) * np.exp(r*T)
#         M = int(S_max / dS)
#         N = int(T / dt)
#         dS = S_max / M
#         dt = T / N

#         S_grid = np.linspace(0, S_max, M+1)
#         V = np.zeros((N+1, M+1))

#         # Terminal payoff: call for S>barrier, else 0
#         payoff = np.maximum(K - S_grid, 0.0)
#         payoff[S_grid <= barrier] = 0.0
#         V[-1,:] = payoff
        
                
#         # Boundaries
#         t_arr = np.linspace(0, T, N+1)
#         for i in range(N+1):
#             tau = T - t_arr[i]
#             V[i, 0]   =0.0
#             V[i, -1]  = K * np.exp(-r*tau)
#             #V[i,-1] = 0

#         # PDE Coeffs
#         j_arr = np.arange(M+1)
#         a = 0.5*dt*(sigma**2 * j_arr**2 - r*j_arr)
#         b = 1.0 - dt*(sigma**2*j_arr**2 + r)
#         c = 0.5*dt*(sigma**2 * j_arr**2 + r*j_arr)

#         # Forward Euler stepping
#         for n in range(N, 0, -1):
#             for j in range(1, M):
#                 V[n-1, j] = a[j]*V[n, j-1] + b[j]*V[n, j] + c[j]*V[n, j+1]
#             # zero out for S <= barrier
#             for j in range(M+1):
#                 if S_grid[j] <= barrier:
#                     V[n-1, j] = 0

#         # Interpolate to get price at S0
#         interp_fn = interp1d(S_grid, V[0,:], kind='linear', fill_value='extrapolate')
#         priceDop = float(interp_fn(S0))
            
#         priceVan, S_gridV, PDE_van = forward_euler_vanilla_put(S0, K, T, r, sigma, dS, dt)
        
#         PDE_dinp = PDE_van - V[0,:]
#         priceDip = priceVan - priceDop
        
#         return priceDip, S_gridV, PDE_dinp
    
#     elif option_type == "up-and-out call":
#         S_max = 2 * max(S0, K) * np.exp(r*T)
#         M = int(S_max / dS)
#         N = int(T / dt)
#         dS = S_max / M
#         dt = T / N

#         S_grid = np.linspace(0, S_max, M+1)
#         V = np.zeros((N+1, M+1))

#         # Terminal payoff: call for S>barrier, else 0
#         payoff = np.maximum(S_grid - K, 0.0)
#         payoff[S_grid >= barrier] = 0.0
#         V[-1,:] = payoff
        
                
#         # Boundaries
#         t_arr = np.linspace(0, T, N+1)
#         for i in range(N+1):
#             tau = T - t_arr[i]
#             V[i, 0]   = 0.0
#             V[i, -1]  = 0.0

#         # PDE Coeffs
#         j_arr = np.arange(M+1)
#         a = 0.5*dt*(sigma**2 * j_arr**2 - r*j_arr)
#         b = 1.0 - dt*(sigma**2*j_arr**2 + r)
#         c = 0.5*dt*(sigma**2 * j_arr**2 + r*j_arr)

#         # Forward Euler stepping
#         for n in range(N, 0, -1):
#             for j in range(1, M):
#                 V[n-1, j] = a[j]*V[n, j-1] + b[j]*V[n, j] + c[j]*V[n, j+1]
#             # zero out for S <= barrier
#             for j in range(M+1):
#                 if S_grid[j] >= barrier:
#                     V[n-1, j] = 0.0

#         # Interpolate to get price at S0
#         interp_fn = interp1d(S_grid, V[0,:], kind='linear', fill_value='extrapolate')
#         priceUoc = float(interp_fn(S0))        
#         return priceUoc, S_grid, V[0,:]
    
#     elif option_type == "up-and-in call":
#         S_max = 2 * max(S0, K) * np.exp(r*T)
#         M = int(S_max / dS)
#         N = int(T / dt)
#         dS = S_max / M
#         dt = T / N

#         S_grid = np.linspace(0, S_max, M+1)
#         V = np.zeros((N+1, M+1))

#         # Terminal payoff: call for S>barrier, else 0
#         payoff = np.maximum(S_grid - K, 0.0)
#         payoff[S_grid >= barrier] = 0.0
#         V[-1,:] = payoff
        
                
#         # Boundaries
#         t_arr = np.linspace(0, T, N+1)
#         for i in range(N+1):
#             tau = T - t_arr[i]
#             V[i, 0]   = 0.0
#             V[i, -1]  = 0.0

#         # PDE Coeffs
#         j_arr = np.arange(M+1)
#         a = 0.5*dt*(sigma**2 * j_arr**2 - r*j_arr)
#         b = 1.0 - dt*(sigma**2*j_arr**2 + r)
#         c = 0.5*dt*(sigma**2 * j_arr**2 + r*j_arr)

#         # Forward Euler stepping
#         for n in range(N, 0, -1):
#             for j in range(1, M):
#                 V[n-1, j] = a[j]*V[n, j-1] + b[j]*V[n, j] + c[j]*V[n, j+1]
#             # zero out for S <= barrier
#             for j in range(M+1):
#                 if S_grid[j] >= barrier:
#                     V[n-1, j] = 0.0

#         # Interpolate to get price at S0
#         interp_fn = interp1d(S_grid, V[0,:], kind='linear', fill_value='extrapolate')
#         priceUoc = float(interp_fn(S0))     

#         priceVan, S_gridV, PDE_van = forward_euler_vanilla_call(S0, K, T, r, sigma, dS, dt)
        
#         PDE_uinc = PDE_van - V[0,:]
#         priceUic = priceVan - priceUoc
        
#         return priceUic, S_gridV, PDE_uinc      
               
#     elif option_type == "up-and-out put":
#         S_max = 2 * max(S0, K) * np.exp(r*T)
#         M = int(S_max / dS)
#         N = int(T / dt)
#         dS = S_max / M
#         dt = T / N

#         S_grid = np.linspace(0, S_max, M+1)
#         V = np.zeros((N+1, M+1))

#         # Terminal payoff: call for S>barrier, else 0
#         payoff = np.maximum(K - S_grid, 0.0)
#         payoff[S_grid >= barrier] = 0.0
#         V[-1,:] = payoff
        
                
#         # Boundaries
#         t_arr = np.linspace(0, T, N+1)
#         for i in range(N+1):
#             tau = T - t_arr[i]
#             V[i, 0]   = 0.0
#             V[i, -1]  = 0.0

#         # PDE Coeffs
#         j_arr = np.arange(M+1)
#         a = 0.5*dt*(sigma**2 * j_arr**2 - r*j_arr)
#         b = 1.0 - dt*(sigma**2*j_arr**2 + r)
#         c = 0.5*dt*(sigma**2 * j_arr**2 + r*j_arr)

#         # Forward Euler stepping
#         for n in range(N, 0, -1):
#             for j in range(1, M):
#                 V[n-1, j] = a[j]*V[n, j-1] + b[j]*V[n, j] + c[j]*V[n, j+1]
#             # zero out for S <= barrier
#             for j in range(M+1):
#                 if S_grid[j] >= barrier:
#                     V[n-1, j] = 0.0

#         # Interpolate to get price at S0
#         interp_fn = interp1d(S_grid, V[0,:], kind='linear', fill_value='extrapolate')
#         priceUop = float(interp_fn(S0))        
#         return priceUop, S_grid, V[0,:]
    
#     elif option_type == "up-and-in put":
#         S_max = 2 * max(S0, K) * np.exp(r*T)
#         M = int(S_max / dS)
#         N = int(T / dt)
#         dS = S_max / M
#         dt = T / N

#         S_grid = np.linspace(0, S_max, M+1)
#         V = np.zeros((N+1, M+1))

#         # Terminal payoff: call for S>barrier, else 0
#         payoff = np.maximum(K - S_grid, 0.0)
#         payoff[S_grid >= barrier] = 0.0
#         V[-1,:] = payoff
        
                
#         # Boundaries
#         t_arr = np.linspace(0, T, N+1)
#         for i in range(N+1):
#             tau = T - t_arr[i]
#             V[i, 0]   = 0.0
#             V[i, -1]  = 0.0

#         # PDE Coeffs
#         j_arr = np.arange(M+1)
#         a = 0.5*dt*(sigma**2 * j_arr**2 - r*j_arr)
#         b = 1.0 - dt*(sigma**2*j_arr**2 + r)
#         c = 0.5*dt*(sigma**2 * j_arr**2 + r*j_arr)

#         # Forward Euler stepping
#         for n in range(N, 0, -1):
#             for j in range(1, M):
#                 V[n-1, j] = a[j]*V[n, j-1] + b[j]*V[n, j] + c[j]*V[n, j+1]
#             # zero out for S <= barrier
#             for j in range(M+1):
#                 if S_grid[j] >= barrier:
#                     V[n-1, j] = 0.0
                    
#         # Interpolate to get price at S0
#         interp_fn = interp1d(S_grid, V[0,:], kind='linear', fill_value='extrapolate')
#         priceUop = float(interp_fn(S0))   
        
#         priceVan, S_gridV, PDE_van = forward_euler_vanilla_put(S0, K, T, r, sigma, dS, dt)
        
#         PDE_uinp = PDE_van - V[0,:]
#         priceUip = priceVan - priceUop
        
#         return priceUip, S_gridV, PDE_uinp      
                            
        
#     return None


###############################################################################
# 1) Vanilla Backward Euler (Call / Put)
###############################################################################
def backward_euler_vanilla_call(S0, K, T, r, sigma, dS, dt):
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


def backward_euler_vanilla_put(S0, K, T, r, sigma, dS, dt):
    """
    Backward Euler PDE for a vanilla European put on [0, S_max].
    Returns: (priceVan, S_grid, V_at_t0).
    """
    S_max = 2 * max(S0, K) * np.exp(r * T)
    M = int(S_max / dS)
    N = int(T / dt)
    dS = S_max / M
    dt = T / N

    S_grid = np.linspace(0, S_max, M + 1)
    V = np.zeros((N + 1, M + 1))

    # Terminal payoff
    V[-1, :] = np.maximum(K - S_grid, 0.0)

    # PDE coefficients
    j_arr = np.arange(M + 1)
    A_ = -0.5 * dt * (sigma**2 * j_arr**2 - r * j_arr)
    B_ =  1.0 + dt * (sigma**2 * j_arr**2 + r)
    C_ = -0.5 * dt * (sigma**2 * j_arr**2 + r * j_arr)

    main_diag = B_[1:M]
    lower_diag = A_[2:M]
    upper_diag = C_[1:M-1]
    T_mat = np.diag(main_diag)
    if M - 2 > 0:
        T_mat += np.diag(lower_diag, k=-1) + np.diag(upper_diag, k=1)
    else:
        T_mat = T_mat.reshape((1, 1))

    # Time-stepping
    t_arr = np.linspace(0, T, N + 1)
    for n in range(N, 0, -1):
        tau = T - t_arr[n - 1]
        # Put boundaries:
        V[n - 1, 0]   = K * np.exp(-r * tau)
        V[n - 1, -1]  = 0.0

        rhs = V[n, 1:M].copy()
        rhs[0]   -= A_[1]     * V[n - 1, 0]
        rhs[-1]  -= C_[M - 1] * V[n - 1, -1]

        V[n - 1, 1:M] = np.linalg.solve(T_mat, rhs)

    interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
    price = float(interp_fn(S0))
    return price, S_grid, V[0, :]


###############################################################################
# 2) Knock-Out (Call / Put) with Backward Euler
###############################################################################
def backward_euler_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type):
    """
    Backward Euler for a knock-out call:
      barrier_type = 'down' => zero out S <= barrier
      barrier_type = 'up'   => zero out S >= barrier
    """
    S_max = 2 * max(S0, K) * np.exp(r * T)
    M = int(S_max / dS)
    N = int(T / dt)
    dS = S_max / M
    dt = T / N
    
    eps = 1e-12  # tolerance threshold

    S_grid = np.linspace(0, S_max, M + 1)
    V = np.zeros((N + 1, M + 1))

    # Terminal payoff for call
    payoff = np.maximum(S_grid - K, 0.0)
    if barrier_type == 'down':
        payoff[S_grid <= barrier] = 0.0
    else:
        payoff[S_grid >= barrier] = 0.0
    V[-1, :] = payoff

    # PDE coefficients
    j_arr = np.arange(M + 1)
    A_ = -0.5 * dt * (sigma**2 * j_arr**2 - r * j_arr)
    B_ =  1.0 + dt * (sigma**2 * j_arr**2 + r)
    C_ = -0.5 * dt * (sigma**2 * j_arr**2 + r * j_arr)

    main_diag = B_[1:M]
    lower_diag = A_[2:M]
    upper_diag = C_[1:M-1]
    T_mat = np.diag(main_diag)
    if M - 2 > 0:
        T_mat += np.diag(lower_diag, k=-1) + np.diag(upper_diag, k=1)
    else:
        T_mat = T_mat.reshape((1, 1))

    # Time-stepping
    t_arr = np.linspace(0, T, N + 1)
    for n in range(N, 0, -1):
        tau = T - t_arr[n - 1]
        V[n - 1, 0]   = 0.0
        V[n - 1, -1]  = S_max - K * np.exp(-r * tau)

        rhs = V[n, 1:M].copy()
        rhs[0]   -= A_[1]     * V[n - 1, 0]
        rhs[-1]  -= C_[M - 1] * V[n - 1, -1]

        V[n - 1, 1:M] = np.linalg.solve(T_mat, rhs)

        # Knock out region
        if barrier_type == 'down':
            V[n - 1, S_grid <= barrier] = 0.0
        else:
            V[n - 1, S_grid >= barrier] = 0.0
            
        V[n - 1, :] = np.where(V[n - 1, :] < eps, 0.0, V[n - 1, :])

    interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
    price_ko = float(interp_fn(S0))
    return price_ko, S_grid, V[0, :]


def backward_euler_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type):
    """
    Backward Euler for a knock-out put:
      barrier_type = 'down' => zero out S <= barrier
      barrier_type = 'up'   => zero out S >= barrier
    """
    S_max = 2 * max(S0, K) * np.exp(r * T)
    M = int(S_max / dS)
    N = int(T / dt)
    dS = S_max / M
    dt = T / N
    
    eps = 1e-12

    S_grid = np.linspace(0, S_max, M + 1)
    V = np.zeros((N + 1, M + 1))

    # Terminal payoff for put
    payoff = np.maximum(K - S_grid, 0.0)
    if barrier_type == 'down':
        payoff[S_grid <= barrier] = 0.0
    else:
        payoff[S_grid >= barrier] = 0.0
    V[-1, :] = payoff

    # PDE coefficients
    j_arr = np.arange(M + 1)
    A_ = -0.5 * dt * (sigma**2 * j_arr**2 - r * j_arr)
    B_ =  1.0 + dt * (sigma**2 * j_arr**2 + r)
    C_ = -0.5 * dt * (sigma**2 * j_arr**2 + r * j_arr)

    main_diag = B_[1:M]
    lower_diag = A_[2:M]
    upper_diag = C_[1:M-1]
    T_mat = np.diag(main_diag)
    if M - 2 > 0:
        T_mat += np.diag(lower_diag, k=-1) + np.diag(upper_diag, k=1)
    else:
        T_mat = T_mat.reshape((1, 1))

    # Time-stepping
    t_arr = np.linspace(0, T, N + 1)
    for n in range(N, 0, -1):
        tau = T - t_arr[n - 1]
        V[n - 1, 0]   = K * np.exp(-r * tau)
        V[n - 1, -1]  = 0.0

        rhs = V[n, 1:M].copy()
        rhs[0]   -= A_[1]     * V[n - 1, 0]
        rhs[-1]  -= C_[M - 1] * V[n - 1, -1]

        V[n - 1, 1:M] = np.linalg.solve(T_mat, rhs)

        # Knock out region
        if barrier_type == 'down':
            V[n - 1, S_grid <= barrier] = 0.0
        else:
            V[n - 1, S_grid >= barrier] = 0.0
        
        V[n - 1, :] = np.where(V[n - 1, :] < eps, 0.0, V[n - 1, :])


    interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
    price_ko = float(interp_fn(S0))
    return price_ko, S_grid, V[0, :]


###############################################################################
# 3) Main Backward Euler Barrier Wrapper
###############################################################################
def backward_euler(S0, K, T, r, sigma, dS, dt, barrier, option_type):
    """
    Main wrapper for backward Euler pricing of barrier options.
    We implement the 'knock-out' PDE directly and use in-out parity:
        knock_in = vanilla - knock_out
    to get the knock-in price.
    """
    # ---------------------------
    # A) DOWN-AND-OUT CALL
    # ---------------------------
    if option_type == "down-and-out call":
        return backward_euler_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down')

    # B) DOWN-AND-IN CALL = vanilla call - down-and-out call
    elif option_type == "down-and-in call":
        priceDOC, Sg_DO, PDE_DO = backward_euler_knock_out_call(
            S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down'
        )
        priceVan, Sg_van, PDE_van = backward_euler_vanilla_call(S0, K, T, r, sigma, dS, dt)
        priceDin = priceVan - priceDOC
        PDE_din  = PDE_van - PDE_DO
        return priceDin, Sg_van, PDE_din

    # C) DOWN-AND-OUT PUT
    elif option_type == "down-and-out put":
        return backward_euler_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down')

    # D) DOWN-AND-IN PUT = vanilla put - down-and-out put
    elif option_type == "down-and-in put":
        priceDOP, Sg_DO, PDE_DO = backward_euler_knock_out_put(
            S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down'
        )
        priceVan, Sg_van, PDE_van = backward_euler_vanilla_put(S0, K, T, r, sigma, dS, dt)
        priceDip = priceVan - priceDOP
        PDE_dip  = PDE_van - PDE_DO
        return priceDip, Sg_van, PDE_dip

    # E) UP-AND-OUT CALL
    elif option_type == "up-and-out call":
        return backward_euler_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up')

    # F) UP-AND-IN CALL = vanilla call - up-and-out call
    elif option_type == "up-and-in call":
        priceUOC, Sg_UO, PDE_UO = backward_euler_knock_out_call(
            S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up'
        )
        priceVan, Sg_van, PDE_van = backward_euler_vanilla_call(S0, K, T, r, sigma, dS, dt)
        priceUic = priceVan - priceUOC
        PDE_uic  = PDE_van - PDE_UO
        return priceUic, Sg_van, PDE_uic

    # G) UP-AND-OUT PUT
    elif option_type == "up-and-out put":
        return backward_euler_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up')

    # H) UP-AND-IN PUT = vanilla put - up-and-out put
    elif option_type == "up-and-in put":
        priceUOP, Sg_UO, PDE_UO = backward_euler_knock_out_put(
            S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up'
        )
        priceVan, Sg_van, PDE_van = backward_euler_vanilla_put(S0, K, T, r, sigma, dS, dt)
        priceUip = priceVan - priceUOP
        PDE_uip  = PDE_van - PDE_UO
        return priceUip, Sg_van, PDE_uip

    # If the option_type is not recognized, return None
    return None

###############################################################################
# A) Vanilla Call / Put with Crank–Nicolson
###############################################################################
# def crank_nicolson_vanilla_call(S0, K, T, r, sigma, dS, dt):
#     """
#     Crank–Nicolson PDE for a vanilla European call on [0, S_max].
#     Returns: (priceCN, S_grid, V_at_t0).
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

#     # 3) Build the CN coefficients for interior nodes j=1,...,M-1
#     #    We'll define the usual "a, b, c" from the forward-Euler approach,
#     #    then split them in half for the Crank–Nicolson system.
#     j_arr = np.arange(M + 1)

#     # From the standard 1D discretization of dV/dt = ...
#     # (matching your forward-Euler 'a, b, c' definitions)
#     a = 0.5 * dt * (sigma**2 * j_arr**2 - r * j_arr)         # multiplies V_{j-1}
#     b = dt * (sigma**2 * j_arr**2 + r)                       # multiplies V_{j}
#     c = 0.5 * dt * (sigma**2 * j_arr**2 + r * j_arr)         # multiplies V_{j+1}

#     # For Crank–Nicolson:
#     #   LHS:  (1 + b[j]/2) on diag,   - a[j]/2 on subdiag,   - c[j]/2 on superdiag
#     #   RHS:  (1 - b[j]/2) on diag,    a[j]/2 on subdiag,     c[j]/2 on superdiag
#     # We'll build these (M-1)x(M-1) matrices for the interior j=1..M-1
#     main_diag_L = np.empty(M - 1)
#     main_diag_R = np.empty(M - 1)
#     sub_diag_L   = np.empty(M - 2)
#     sub_diag_R   = np.empty(M - 2)
#     super_diag_L = np.empty(M - 2)
#     super_diag_R = np.empty(M - 2)

#     for j in range(1, M):
#         main_diag_L[j - 1] = 1 + b[j] / 2
#         main_diag_R[j - 1] = 1 - b[j] / 2

#     for j in range(1, M - 1):
#         sub_diag_L[j - 1] = -0.5 * a[j + 1]
#         sub_diag_R[j - 1] =  0.5 * a[j + 1]
#         super_diag_L[j - 1] = -0.5 * c[j]
#         super_diag_R[j - 1] =  0.5 * c[j]

#     # We'll construct the LHS and RHS matrices (both tridiagonal)
#     # Then each time step we solve: LHS * V^{n-1}_int = RHS * V^n_int
#     # 'int' means the interior nodes [1..M-1].
#     import scipy.sparse as sp
#     import scipy.sparse.linalg as spla

#     # Helper function to build a tridiagonal sparse matrix
#     def build_tridiag(main, sub, super_):
#         return sp.diags(
#             diagonals=[sub, main, super_],
#             offsets=[-1, 0, 1],
#             shape=(len(main), len(main)),
#             format='csr'
#         )

#     LHS_mat = build_tridiag(main_diag_L, sub_diag_L, super_diag_L)
#     RHS_mat = build_tridiag(main_diag_R, sub_diag_R, super_diag_R)

#     # 4) Time stepping from n=N down to n=1
#     t_arr = np.linspace(0, T, N + 1)
#     for n in range(N, 0, -1):
#         tau = T - t_arr[n - 1]

#         # Boundary conditions at the new time level (n-1):
#         # For a call:
#         V[n - 1, 0]   = 0.0
#         V[n - 1, -1]  = S_max - K * np.exp(-r * tau)

#         # Build the RHS vector = RHS_mat * V[n, 1..M-1], plus boundary adjustments
#         rhs_vec = RHS_mat.dot(V[n, 1:M])

#         # Now incorporate known boundaries into rhs_vec:
#         # sub_diag_R[0] multiplies V[n,0], super_diag_R[-1] multiplies V[n,M]
#         # but for CN we need half-coeffs, etc.
#         # Carefully handle the first and last interior points:
#         j_first = 1
#         j_last  = M - 1

#         # If there's more than one interior node:
#         if (M - 1) >= 1:
#             # sub diag => index 0 in sub_diag_R corresponds to j=2 in full
#             # but for j=1 we incorporate a boundary for sub
#             rhs_vec[0] -= sub_diag_R[0] * V[n - 1, 0]  # boundary at j=0
#             rhs_vec[-1] -= super_diag_R[-1] * V[n - 1, M]  # boundary at j=M

#         # Solve LHS_mat * V[n-1,1..M-1] = rhs_vec
#         V[n - 1, 1:M] = spla.spsolve(LHS_mat, rhs_vec)

#     # 5) Interpolate to get the option price at S0
#     interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
#     priceCN = float(interp_fn(S0))
#     return priceCN, S_grid, V[0, :]


# def crank_nicolson_vanilla_put(S0, K, T, r, sigma, dS, dt):
#     """
#     Crank–Nicolson PDE for a vanilla European put on [0, S_max].
#     Returns: (priceCN, S_grid, V_at_t0).
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

#     # Build CN coefficients (same pattern as the call)
#     j_arr = np.arange(M + 1)
#     a = 0.5 * dt * (sigma**2 * j_arr**2 - r * j_arr)
#     b = dt * (sigma**2 * j_arr**2 + r)
#     c = 0.5 * dt * (sigma**2 * j_arr**2 + r * j_arr)

#     main_diag_L = np.empty(M - 1)
#     main_diag_R = np.empty(M - 1)
#     sub_diag_L   = np.empty(M - 2)
#     sub_diag_R   = np.empty(M - 2)
#     super_diag_L = np.empty(M - 2)
#     super_diag_R = np.empty(M - 2)

#     for j in range(1, M):
#         main_diag_L[j - 1] = 1 + b[j] / 2
#         main_diag_R[j - 1] = 1 - b[j] / 2

#     for j in range(1, M - 1):
#         sub_diag_L[j - 1]   = -0.5 * a[j + 1]
#         sub_diag_R[j - 1]   =  0.5 * a[j + 1]
#         super_diag_L[j - 1] = -0.5 * c[j]
#         super_diag_R[j - 1] =  0.5 * c[j]

#     import scipy.sparse as sp
#     import scipy.sparse.linalg as spla

#     def build_tridiag(main, sub, super_):
#         return sp.diags(
#             diagonals=[sub, main, super_],
#             offsets=[-1, 0, 1],
#             shape=(len(main), len(main)),
#             format='csr'
#         )

#     LHS_mat = build_tridiag(main_diag_L, sub_diag_L, super_diag_L)
#     RHS_mat = build_tridiag(main_diag_R, sub_diag_R, super_diag_R)

#     # Time stepping
#     t_arr = np.linspace(0, T, N + 1)
#     for n in range(N, 0, -1):
#         tau = T - t_arr[n - 1]
#         # Put boundary conditions
#         V[n - 1, 0]   = K * np.exp(-r * tau)  # deep in-the-money at S=0
#         V[n - 1, -1]  = 0.0                  # worthless at large S

#         rhs_vec = RHS_mat.dot(V[n, 1:M])

#         # Adjust for boundaries
#         if (M - 1) >= 1:
#             rhs_vec[0]   -= sub_diag_R[0]   * V[n - 1, 0]
#             rhs_vec[-1]  -= super_diag_R[-1]* V[n - 1, M]

#         V[n - 1, 1:M] = spla.spsolve(LHS_mat, rhs_vec)

#     interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
#     priceCN = float(interp_fn(S0))
#     return priceCN, S_grid, V[0, :]

# ###############################################################################
# # B) Knock-Out (Call / Put) with Crank–Nicolson
# ###############################################################################
# def crank_nicolson_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type):
#     """
#     Crank–Nicolson PDE for a knock-out call:
#       barrier_type = 'down' => zero out for S <= barrier
#       barrier_type = 'up'   => zero out for S >= barrier
#     """
#     S_max = 2 * max(S0, K) * np.exp(r * T)
#     M = int(S_max / dS)
#     N = int(T / dt)
#     dS = S_max / M
#     dt = T / N

#     S_grid = np.linspace(0, S_max, M + 1)
#     V = np.zeros((N + 1, M + 1))

#     # 1) Terminal payoff for a call
#     payoff = np.maximum(S_grid - K, 0.0)
#     if barrier_type == 'down':
#         payoff[S_grid <= barrier] = 0.0
#     else:
#         payoff[S_grid >= barrier] = 0.0
#     V[-1, :] = payoff

#     # 2) Build the CN matrices (same as vanilla call)
#     j_arr = np.arange(M + 1)
#     a = 0.5 * dt * (sigma**2 * j_arr**2 - r * j_arr)
#     b = dt * (sigma**2 * j_arr**2 + r)
#     c = 0.5 * dt * (sigma**2 * j_arr**2 + r * j_arr)

#     # Construct LHS and RHS
#     main_diag_L = np.empty(M - 1)
#     main_diag_R = np.empty(M - 1)
#     sub_diag_L   = np.empty(M - 2)
#     sub_diag_R   = np.empty(M - 2)
#     super_diag_L = np.empty(M - 2)
#     super_diag_R = np.empty(M - 2)

#     for j in range(1, M):
#         main_diag_L[j - 1] = 1 + b[j] / 2
#         main_diag_R[j - 1] = 1 - b[j] / 2

#     for j in range(1, M - 1):
#         sub_diag_L[j - 1]   = -0.5 * a[j + 1]
#         sub_diag_R[j - 1]   =  0.5 * a[j + 1]
#         super_diag_L[j - 1] = -0.5 * c[j]
#         super_diag_R[j - 1] =  0.5 * c[j]

#     import scipy.sparse as sp
#     import scipy.sparse.linalg as spla

#     def build_tridiag(main, sub, super_):
#         return sp.diags(
#             diagonals=[sub, main, super_],
#             offsets=[-1, 0, 1],
#             shape=(len(main), len(main)),
#             format='csr'
#         )

#     LHS_mat = build_tridiag(main_diag_L, sub_diag_L, super_diag_L)
#     RHS_mat = build_tridiag(main_diag_R, sub_diag_R, super_diag_R)

#     # 3) Time stepping
#     t_arr = np.linspace(0, T, N + 1)
#     for n in range(N, 0, -1):
#         tau = T - t_arr[n - 1]
#         # Call boundary conditions
#         V[n - 1, 0]   = 0.0
#         V[n - 1, -1]  = S_max - K * np.exp(-r * tau)

#         # Build RHS
#         rhs_vec = RHS_mat.dot(V[n, 1:M])
#         if (M - 1) >= 1:
#             rhs_vec[0]   -= sub_diag_R[0]   * V[n - 1, 0]
#             rhs_vec[-1]  -= super_diag_R[-1]* V[n - 1, M]

#         # Solve
#         V[n - 1, 1:M] = spla.spsolve(LHS_mat, rhs_vec)

#         # Knock out region
#         if barrier_type == 'down':
#             V[n - 1, S_grid <= barrier] = 0.0
#         else:
#             V[n - 1, S_grid >= barrier] = 0.0

#     # 4) Interpolate to get price at S0
#     interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
#     price_ko = float(interp_fn(S0))
#     return price_ko, S_grid, V[0, :]


# def crank_nicolson_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type):
#     """
#     Crank–Nicolson PDE for a knock-out put:
#       barrier_type = 'down' => zero out for S <= barrier
#       barrier_type = 'up'   => zero out for S >= barrier
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
#         payoff[S_grid <= barrier] = 0.0
#     else:
#         payoff[S_grid >= barrier] = 0.0
#     V[-1, :] = payoff

#     # Build CN matrices (similar to vanilla put)
#     j_arr = np.arange(M + 1)
#     a = 0.5 * dt * (sigma**2 * j_arr**2 - r * j_arr)
#     b = dt * (sigma**2 * j_arr**2 + r)
#     c = 0.5 * dt * (sigma**2 * j_arr**2 + r * j_arr)

#     main_diag_L = np.empty(M - 1)
#     main_diag_R = np.empty(M - 1)
#     sub_diag_L   = np.empty(M - 2)
#     sub_diag_R   = np.empty(M - 2)
#     super_diag_L = np.empty(M - 2)
#     super_diag_R = np.empty(M - 2)

#     for j in range(1, M):
#         main_diag_L[j - 1] = 1 + b[j] / 2
#         main_diag_R[j - 1] = 1 - b[j] / 2

#     for j in range(1, M - 1):
#         sub_diag_L[j - 1]   = -0.5 * a[j + 1]
#         sub_diag_R[j - 1]   =  0.5 * a[j + 1]
#         super_diag_L[j - 1] = -0.5 * c[j]
#         super_diag_R[j - 1] =  0.5 * c[j]

#     import scipy.sparse as sp
#     import scipy.sparse.linalg as spla

#     def build_tridiag(main, sub, super_):
#         return sp.diags(
#             diagonals=[sub, main, super_],
#             offsets=[-1, 0, 1],
#             shape=(len(main), len(main)),
#             format='csr'
#         )

#     LHS_mat = build_tridiag(main_diag_L, sub_diag_L, super_diag_L)
#     RHS_mat = build_tridiag(main_diag_R, sub_diag_R, super_diag_R)

#     # Time stepping
#     t_arr = np.linspace(0, T, N + 1)
#     for n in range(N, 0, -1):
#         tau = T - t_arr[n - 1]
#         # Put boundary conditions
#         V[n - 1, 0]   = K * np.exp(-r * tau)
#         V[n - 1, -1]  = 0.0

#         rhs_vec = RHS_mat.dot(V[n, 1:M])
#         if (M - 1) >= 1:
#             rhs_vec[0]   -= sub_diag_R[0]   * V[n - 1, 0]
#             rhs_vec[-1]  -= super_diag_R[-1]* V[n - 1, M]

#         V[n - 1, 1:M] = spla.spsolve(LHS_mat, rhs_vec)

#         # Knock out region
#         if barrier_type == 'down':
#             V[n - 1, S_grid <= barrier] = 0.0
#         else:
#             V[n - 1, S_grid >= barrier] = 0.0

#     interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
#     price_ko = float(interp_fn(S0))
#     return price_ko, S_grid, V[0, :]


# ###############################################################################
# # C) Main Crank–Nicolson Barrier Wrapper
# ###############################################################################
# def crank_nicolson(S0, K, T, r, sigma, dS, dt, barrier, option_type):
#     """
#     Main wrapper for Crank–Nicolson pricing of barrier options.
#     We implement PDE directly for 'knock-out' and use in-out parity:
#         knock_in = vanilla - knock_out
#     to get the knock-in price.
    
#     Supported option_type:
#       "down-and-out call",  "down-and-in call",
#       "down-and-out put",   "down-and-in put",
#       "up-and-out call",    "up-and-in call",
#       "up-and-out put",     "up-and-in put".
#     """
#     # ---------------------------
#     # A) DOWN-AND-OUT CALL
#     # ---------------------------
#     if option_type == "down-and-out call":
#         return crank_nicolson_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down')

#     # B) DOWN-AND-IN CALL = vanilla call - down-and-out call
#     elif option_type == "down-and-in call":
#         priceDOC, Sg_DO, PDE_DO = crank_nicolson_knock_out_call(
#             S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down'
#         )
#         priceVan, Sg_van, PDE_van = crank_nicolson_vanilla_call(S0, K, T, r, sigma, dS, dt)
#         priceDin = priceVan - priceDOC
#         PDE_din  = PDE_van - PDE_DO
#         return priceDin, Sg_van, PDE_din

#     # C) DOWN-AND-OUT PUT
#     elif option_type == "down-and-out put":
#         return crank_nicolson_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down')

#     # D) DOWN-AND-IN PUT = vanilla put - down-and-out put
#     elif option_type == "down-and-in put":
#         priceDOP, Sg_DO, PDE_DO = crank_nicolson_knock_out_put(
#             S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down'
#         )
#         priceVan, Sg_van, PDE_van = crank_nicolson_vanilla_put(S0, K, T, r, sigma, dS, dt)
#         priceDip = priceVan - priceDOP
#         PDE_dip  = PDE_van - PDE_DO
#         return priceDip, Sg_van, PDE_dip

#     # E) UP-AND-OUT CALL
#     elif option_type == "up-and-out call":
#         return crank_nicolson_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up')

#     # F) UP-AND-IN CALL = vanilla call - up-and-out call
#     elif option_type == "up-and-in call":
#         priceUOC, Sg_UO, PDE_UO = crank_nicolson_knock_out_call(
#             S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up'
#         )
#         priceVan, Sg_van, PDE_van = crank_nicolson_vanilla_call(S0, K, T, r, sigma, dS, dt)
#         priceUic = priceVan - priceUOC
#         PDE_uic  = PDE_van - PDE_UO
#         return priceUic, Sg_van, PDE_uic

#     # G) UP-AND-OUT PUT
#     elif option_type == "up-and-out put":
#         return crank_nicolson_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up')

#     # H) UP-AND-IN PUT = vanilla put - up-and-out put
#     elif option_type == "up-and-in put":
#         priceUOP, Sg_UO, PDE_UO = crank_nicolson_knock_out_put(
#             S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up'
#         )
#         priceVan, Sg_van, PDE_van = crank_nicolson_vanilla_put(S0, K, T, r, sigma, dS, dt)
#         priceUip = priceVan - priceUOP
#         PDE_uip  = PDE_van - PDE_UO
#         return priceUip, Sg_van, PDE_uip

#     # If not recognized, return None
#     return None

###############################################################################
# A) Vanilla Call / Put with Crank–Nicolson
###############################################################################
import scipy.sparse as sp
import scipy.sparse.linalg as spla
def crank_nicolson_vanilla_call(S0, K, T, r, sigma, dS, dt):
    """
    Crank–Nicolson PDE for a vanilla European call on [0, S_max].
    Returns: (priceCN, S_grid, V_at_t0)
    """
    # 1) Grid setup
    S_max = 2 * max(S0, K) * np.exp(r * T)
    M = int(S_max / dS)
    N = int(T / dt)
    dS = S_max / M
    dt = T / N

    S_grid = np.linspace(0, S_max, M + 1)
    V = np.zeros((N + 1, M + 1))

    # 2) Terminal payoff: call = max(S-K,0)
    V[-1, :] = np.maximum(S_grid - K, 0.0)

    # 3) Build the CN coefficients for interior nodes j=1,...,M-1
    j_arr = np.arange(M + 1)
    a = 0.5 * dt * (sigma**2 * j_arr**2 - r * j_arr)  # multiplies V[j-1]
    b = dt * (sigma**2 * j_arr**2 + r)                # multiplies V[j]
    c = 0.5 * dt * (sigma**2 * j_arr**2 + r * j_arr)    # multiplies V[j+1]

    # For Crank–Nicolson:
    # LHS:  diag: (1 + b[j]/2), subdiag: -a[j]/2, superdiag: -c[j]/2
    # RHS:  diag: (1 - b[j]/2), subdiag:  a[j]/2, superdiag:  c[j]/2
    main_diag_L = np.empty(M - 1)
    main_diag_R = np.empty(M - 1)
    sub_diag_L   = np.empty(M - 2)
    sub_diag_R   = np.empty(M - 2)
    super_diag_L = np.empty(M - 2)
    super_diag_R = np.empty(M - 2)

    for j in range(1, M):
        main_diag_L[j - 1] = 1 + b[j] / 2
        main_diag_R[j - 1] = 1 - b[j] / 2

    for j in range(1, M - 1):
        sub_diag_L[j - 1] = -0.5 * a[j + 1]
        sub_diag_R[j - 1] =  0.5 * a[j + 1]
        super_diag_L[j - 1] = -0.5 * c[j]
        super_diag_R[j - 1] =  0.5 * c[j]

    # Build tridiagonal matrices for interior nodes (i=1,..,M-1)
    def build_tridiag(main, sub, sup):
        return sp.diags([sub, main, sup], offsets=[-1, 0, 1], format='csr')
    
    LHS_mat = build_tridiag(main_diag_L, sub_diag_L, super_diag_L)
    RHS_mat = build_tridiag(main_diag_R, sub_diag_R, super_diag_R)

    # 4) Time stepping from n = N down to n = 1.
    t_arr = np.linspace(0, T, N + 1)
    for n in range(N, 0, -1):
        tau = T - t_arr[n - 1]
        # Set boundary conditions for call:
        V[n - 1, 0] = 0.0
        V[n - 1, -1] = S_max - K * np.exp(-r * tau)
        
        # Build RHS vector from interior nodes at time level n.
        rhs_vec = RHS_mat.dot(V[n, 1:M])
        # Incorporate boundary adjustments using an aux (auxiliary) variable:
        aux = np.zeros(M - 1)
        # For the first interior node, add contribution from S=0:
        aux[0] = sub_diag_R[0] * V[n - 1, 0]  # note: sub_diag_R = 0.5*a[j+1]
        # For the last interior node, add contribution from S=S_max:
        aux[-1] = super_diag_R[-1] * V[n - 1, M]
        rhs_vec -= aux
        
        # Solve for interior nodes at time level n-1.
        V[n - 1, 1:M] = spla.spsolve(LHS_mat, rhs_vec)
    # 5) Interpolate to get the option price at S0.
    interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
    priceCN = float(interp_fn(S0))
    return priceCN, S_grid, V[0, :]


def crank_nicolson_vanilla_put(S0, K, T, r, sigma, dS, dt):
    """
    Crank–Nicolson PDE for a vanilla European put on [0, S_max].
    Returns: (priceCN, S_grid, V_at_t0).
    """
    S_max = 2 * max(S0, K) * np.exp(r * T)
    M = int(S_max / dS)
    N = int(T / dt)
    dS = S_max / M
    dt = T / N

    S_grid = np.linspace(0, S_max, M + 1)
    V = np.zeros((N + 1, M + 1))
    
    # Terminal payoff: put = max(K - S, 0)
    V[-1, :] = np.maximum(K - S_grid, 0.0)
    
    # Set boundary conditions for a put:
    for j in range(N+1):
        tau = T - j * dt
        V[j, 0] = K * np.exp(-r * tau)  # at S = 0
        V[j, -1] = 0.0                  # at S = S_max

    j_arr = np.arange(M + 1)
    a = 0.5 * dt * (sigma**2 * j_arr**2 - r * j_arr)
    b = dt * (sigma**2 * j_arr**2 + r)
    c = 0.5 * dt * (sigma**2 * j_arr**2 + r * j_arr)
    
    main_diag_L = np.empty(M - 1)
    main_diag_R = np.empty(M - 1)
    sub_diag_L   = np.empty(M - 2)
    sub_diag_R   = np.empty(M - 2)
    super_diag_L = np.empty(M - 2)
    super_diag_R = np.empty(M - 2)
    
    for j in range(1, M):
        main_diag_L[j - 1] = 1 + b[j] / 2
        main_diag_R[j - 1] = 1 - b[j] / 2
    for j in range(1, M - 1):
        sub_diag_L[j - 1] = -0.5 * a[j + 1]
        sub_diag_R[j - 1] =  0.5 * a[j + 1]
        super_diag_L[j - 1] = -0.5 * c[j]
        super_diag_R[j - 1] =  0.5 * c[j]
        
    def build_tridiag(main, sub, sup):
        return sp.diags([sub, main, sup], offsets=[-1, 0, 1], format='csr')
    
    LHS_mat = build_tridiag(main_diag_L, sub_diag_L, super_diag_L)
    RHS_mat = build_tridiag(main_diag_R, sub_diag_R, super_diag_R)
    
    t_arr = np.linspace(0, T, N + 1)
    for n in range(N, 0, -1):
        tau = T - t_arr[n - 1]
        V[n - 1, 0] = K * np.exp(-r * tau)
        V[n - 1, -1] = 0.0
        
        rhs_vec = RHS_mat.dot(V[n, 1:M])
        aux = np.zeros(M - 1)
        aux[0] = sub_diag_R[0] * V[n - 1, 0]
        aux[-1] = super_diag_R[-1] * V[n - 1, M]
        rhs_vec -= aux
        
        V[n - 1, 1:M] = spla.spsolve(LHS_mat, rhs_vec)
    
    interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
    priceCN = float(interp_fn(S0))
    return priceCN, S_grid, V[0, :]

###############################################################################
# B) Knock–Out Options with Crank–Nicolson (Call / Put)
###############################################################################
def crank_nicolson_knock_out(S0, K, T, r, sigma, dS, dt, barrier, barrier_type, option_type):
    """
    Crank–Nicolson PDE for a knock–out option.
    
    barrier_type: 'down' => zero out for S <= barrier,
                  'up'   => zero out for S >= barrier.
    option_type: "Call" or "Put".
    
    The terminal payoff is modified by zeroing values in the barrier region.
    During the time stepping, after solving the system we enforce the barrier
    by setting V = 0 for S in the knocked–out region.
    """
    S_max = 2 * max(S0, K) * np.exp(r * T)
    M = int(S_max / dS)
    N = int(T / dt)
    dS = S_max / M
    dt = T / N

    S_grid = np.linspace(0, S_max, M + 1)
    V = np.zeros((N + 1, M + 1))
    
    # 1) Terminal payoff, modified by barrier:
    if option_type.lower() == "call":
        payoff = np.maximum(S_grid - K, 0.0)
    else:
        payoff = np.maximum(K - S_grid, 0.0)
    if barrier_type == 'down':
        payoff[S_grid <= barrier] = 0.0
    else:
        payoff[S_grid >= barrier] = 0.0
    V[-1, :] = payoff

    # 2) Boundary conditions (same as vanilla)
    for j in range(N+1):
        tau = T - j*dt
        if option_type.lower() == "call":
            V[j, 0] = 0.0
            V[j, -1] = S_max - K * np.exp(-r * tau)
        else:
            V[j, 0] = K * np.exp(-r * tau)
            V[j, -1] = 0.0

    # 3) Build CN coefficients (same as vanilla)
    j_arr = np.arange(M + 1)
    a = 0.5 * dt * (sigma**2 * j_arr**2 - r * j_arr)
    b = dt * (sigma**2 * j_arr**2 + r)
    c = 0.5 * dt * (sigma**2 * j_arr**2 + r * j_arr)
    
    main_diag_L = np.empty(M - 1)
    main_diag_R = np.empty(M - 1)
    sub_diag_L   = np.empty(M - 2)
    sub_diag_R   = np.empty(M - 2)
    super_diag_L = np.empty(M - 2)
    super_diag_R = np.empty(M - 2)
    
    for j in range(1, M):
        main_diag_L[j - 1] = 1 + b[j] / 2
        main_diag_R[j - 1] = 1 - b[j] / 2
    for j in range(1, M - 1):
        sub_diag_L[j - 1] = -0.5 * a[j + 1]
        sub_diag_R[j - 1] =  0.5 * a[j + 1]
        super_diag_L[j - 1] = -0.5 * c[j]
        super_diag_R[j - 1] =  0.5 * c[j]
        
    def build_tridiag(main, sub, sup):
        return sp.diags([sub, main, sup], offsets=[-1, 0, 1], format='csr')
    
    LHS_mat = build_tridiag(main_diag_L, sub_diag_L, super_diag_L)
    RHS_mat = build_tridiag(main_diag_R, sub_diag_R, super_diag_R)
    
    # 4) Time stepping (backward in time) with aux for boundary adjustments.
    t_arr = np.linspace(0, T, N + 1)
    for n in range(N, 0, -1):
        tau = T - t_arr[n - 1]
        if option_type.lower() == "call":
            V[n - 1, 0] = 0.0
            V[n - 1, -1] = S_max - K * np.exp(-r * tau)
        else:
            V[n - 1, 0] = K * np.exp(-r * tau)
            V[n - 1, -1] = 0.0
        
        rhs_vec = RHS_mat.dot(V[n, 1:M])
        aux = np.zeros(M - 1)
        aux[0] = sub_diag_R[0] * V[n - 1, 0]
        aux[-1] = super_diag_R[-1] * V[n - 1, M]
        rhs_vec -= aux
        
        V[n - 1, 1:M] = spla.spsolve(LHS_mat, rhs_vec)
        
        # 5) Enforce barrier condition at each time step:
        if barrier_type == 'down':
            V[n - 1, S_grid <= barrier] = 0.0
        else:
            V[n - 1, S_grid >= barrier] = 0.0
    # 6) Interpolate to get price at S0.
    interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
    price_ko = float(interp_fn(S0))
    return price_ko, S_grid, V[0, :]

###############################################################################
# C) Main Crank–Nicolson Barrier Wrapper Using In–Out Parity
###############################################################################
def crank_nicolson_barrier(S0, K, T, r, sigma, dS, dt, barrier, option_type):
    """
    Main wrapper for Crank–Nicolson pricing of barrier options.
    
    option_type should be one of:
      "down-and-out call",  "down-and-in call",
      "down-and-out put",   "down-and-in put",
      "up-and-out call",    "up-and-in call",
      "up-and-out put",     "up-and-in put".
    
    For knock–in options, we compute the vanilla price and subtract the knock–out price.
    """
    # Determine if vanilla is Call or Put
    if "call" in option_type.lower():
        vanilla_func = crank_nicolson_vanilla_call
    else:
        vanilla_func = crank_nicolson_vanilla_put

    # For knock–out options:
    if "down-and-out" in option_type.lower():
        barrier_type = 'down'
        price_ko, S_grid, PDE_ko = crank_nicolson_knock_out(S0, K, T, r, sigma, dS, dt,
                                                             barrier, barrier_type,
                                                             option_type.split()[-1])
        # For knock–in:
        if "in" in option_type.lower():
            price_van, S_grid, PDE_van = vanilla_func(S0, K, T, r, sigma, dS, dt)
            price_ki = price_van - price_ko
            PDE_ki = PDE_van - PDE_ko
            return price_ki, S_grid, PDE_ki
        else:
            return price_ko, S_grid, PDE_ko
    elif "up-and-out" in option_type.lower():
        barrier_type = 'up'
        price_ko, S_grid, PDE_ko = crank_nicolson_knock_out(S0, K, T, r, sigma, dS, dt,
                                                             barrier, barrier_type,
                                                             option_type.split()[-1])
        if "in" in option_type.lower():
            price_van, S_grid, PDE_van = vanilla_func(S0, K, T, r, sigma, dS, dt)
            price_ki = price_van - price_ko
            PDE_ki = PDE_van - PDE_ko
            return price_ki, S_grid, PDE_ki
        else:
            return price_ko, S_grid, PDE_ko
    else:
        return None
################################################################################
# 5) The Streamlit app
################################################################################
def app():
    st.title("Barrier Options: PDE vs Analytical Barrier Formula")

    # Sidebar inputs
    S0 = st.sidebar.number_input("Spot Price (S0)", value=100.0, step=1.0)
    K  = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0)
    T  = st.sidebar.number_input("Time to Maturity (T)", value=1.0, step=0.00001)
    r  = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)
    q  = st.sidebar.number_input("Dividend Yield (q)", value=0.00, step=0.01)
    sigma = st.sidebar.number_input("Volatility (sigma)", value=0.2, step=0.01)
    barrier = st.sidebar.number_input("Barrier", value=80.0, step=1.0)
    dS = st.sidebar.number_input("Space Step (dS)", value=1.0, step=0.1)
    dt = st.sidebar.number_input("Time Step (dt)", value=0.001, step=0.001)
    option_type = st.sidebar.selectbox(
    "Option Type",
    [
        "down-and-in call",
        "down-and-out call",
        "down-and-in put",
        "down-and-out put",
        "up-and-in call",
        "up-and-out call",
        "up-and-in put",
        "up-and-out put",
    ])
    numerical_method = st.sidebar.selectbox("Numerical method", ("Forward Euler", "Backward Euler", "Crank-Nicolson"))


    if numerical_method == "Forward Euler":
    # 1) PDE solution for down-and-in
        #priceDin, S_grid, PDE_din = forward_euler_down_in_call(S0, K, T, r, sigma, dS, dt, barrier)
        priceSol, S_grid, PDE_sol = forward_euler(S0, K, T, r, sigma, dS, dt, barrier, option_type)
        

        st.write(f"**PDE price at S0** = {priceSol:.4f}")

        # 2) Evaluate the closed-form formula over the same S_grid
        #    for "down-and-in call"
        analytic_vals = []
        for s in S_grid:
            val = barrier_option_price(s, K, T, r, q, sigma, barrier, option_type)
            # Some formulas may return None if input is out of domain logic.
            # We'll assume 0 if None is returned.
            if val is None or val < 0:
                val = 0.0
            analytic_vals.append(val)
            
        new_pde = PDE_sol[1:]
        # 3) Plot PDE vs Analytical
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=S_grid, 
            #y=PDE_sol, 
            y = new_pde,
            mode="markers+lines", 
            name="PDE Solution"
        ))
        fig.add_trace(go.Scatter(
            x=S_grid,
            y=analytic_vals,
            mode="lines",
            name="Analytical "+option_type
        ))
        fig.update_layout(
            title= option_type+" vs Analytical",
            xaxis_title="Stock Price (S)",
            yaxis_title="Option Value"
        )
        st.plotly_chart(fig)
        
    elif numerical_method == "Backward Euler":
    # 1) PDE solution for down-and-in
        #priceDin, S_grid, PDE_din = forward_euler_down_in_call(S0, K, T, r, sigma, dS, dt, barrier)
        priceSol, S_grid, PDE_sol = backward_euler(S0, K, T, r, sigma, dS, dt, barrier, option_type)
        

        st.write(f"**PDE price at S0** = {priceSol:.4f}")

        # 2) Evaluate the closed-form formula over the same S_grid
        #    for "down-and-in call"
        analytic_vals = []
        for s in S_grid:
            val = barrier_option_price(s, K, T, r, q, sigma, barrier, option_type)
            # Some formulas may return None if input is out of domain logic.
            # We'll assume 0 if None is returned.
            if val is None or val < 0:
                val = 0.0
            analytic_vals.append(val)
            
        new_pde = PDE_sol[2:]
        # 3) Plot PDE vs Analytical
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=S_grid, 
            #y=PDE_sol, 
            y = new_pde,
            mode="markers+lines", 
            name="PDE Solution"
        ))
        fig.add_trace(go.Scatter(
            x=S_grid,
            y=analytic_vals,
            mode="lines",
            name="Analytical "+option_type
        ))
        fig.update_layout(
            title= option_type+" vs Analytical",
            xaxis_title="Stock Price (S)",
            yaxis_title="Option Value"
        )
        st.plotly_chart(fig)
        
    elif numerical_method == "Crank-Nicolson":
    # 1) PDE solution for down-and-in
        #priceDin, S_grid, PDE_din = forward_euler_down_in_call(S0, K, T, r, sigma, dS, dt, barrier)
        priceSol, S_grid, PDE_sol = crank_nicolson_barrier(S0, K, T, r, sigma, dS, dt, barrier, option_type)
        

        st.write(f"**PDE price at S0** = {priceSol:.4f}")

        # 2) Evaluate the closed-form formula over the same S_grid
        #    for "down-and-in call"
        analytic_vals = []
        for s in S_grid:
            val = barrier_option_price(s, K, T, r, q, sigma, barrier, option_type)
            # Some formulas may return None if input is out of domain logic.
            # We'll assume 0 if None is returned.
            if val is None or val < 0:
                val = 0.0
            analytic_vals.append(val)
            
        new_pde = PDE_sol[1:]
        # 3) Plot PDE vs Analytical
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=S_grid, 
            #y=PDE_sol, 
            y = new_pde,
            mode="markers+lines", 
            name="PDE Solution"
        ))
        fig.add_trace(go.Scatter(
            x=S_grid,
            y=analytic_vals,
            mode="lines",
            name="Analytical "+option_type
        ))
        fig.update_layout(
            title= option_type+" vs Analytical",
            xaxis_title="Stock Price (S)",
            yaxis_title="Option Value"
        )
        st.plotly_chart(fig)
        
if __name__ == "__main__":
    app()

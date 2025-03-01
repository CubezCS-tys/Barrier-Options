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

# def calc_y(H, S0, K, T, sigma, r, q):
#     """
#     y = [ln(H^2/(S0*K)) / (sigma*sqrt(T))] + λ * sigma * sqrt(T)
#     """
#     lam = calc_lambda(r, q, sigma)
#     return (np.log((H**2)/(S0*K)) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

# def calc_x1(S0, H, T, sigma, r, q):
#     """
#     x1 = ln(S0/H)/(sigma*sqrt(T)) + λ*sigma*sqrt(T)
#     """
#     lam = calc_lambda(r, q, sigma)
#     return (np.log(S0/H) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

# def calc_y1(S0, H, T, sigma, r, q):
#     """
#     y1 = ln(H/S0)/(sigma*sqrt(T)) + λ*sigma*sqrt(T)
#     """
#     lam = calc_lambda(r, q, sigma)
#     return (np.log(H/S0) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

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

# def barrier_option_price(S0, K, T, r, q, sigma, H, option_type):
#     """
#     Returns the price of a barrier option (various knock-in/out types).
#     Matches standard formulas from texts like Hull, with care to keep
#     exponents and sign conventions correct.
#     """
#     x1 = calc_x1(S0, H, T, sigma, r, q)
#     y1 = calc_y1(S0, H, T, sigma, r, q)
#     c = calc_c(S0, K, r, q, sigma, T)
#     p = calc_p(S0, K, r, q, sigma, T)
#     lam = calc_lambda(r, q, sigma)
#     y  = calc_y(barrier, S0, K, T, sigma, r, q)

#     # --------------------------------
#     # Down-and-in Call
#     # --------------------------------
    
#     if option_type == 'down-and-in call' and barrier <= K and np.any(S0) <= barrier:
#         vanilla = black_scholes(S0, K, T, r, sigma, "Call")
#         return vanilla
    
#     elif option_type == 'down-and-in call' and barrier <= K:
#         # cdi, for H <= K
#         cdi = (S0 * np.exp(-q*T) * (H/S0)**(2*lam) * norm.cdf(y)
#                - K * np.exp(-r*T) * (H/S0)**(2*lam - 2)
#                  * norm.cdf(y - sigma*np.sqrt(T)))
#         return cdi

#     elif option_type == 'down-and-in call' and H >= K:
#         # cdi = c - cdo. So we compute cdo from the standard expression
#         # cdo = ...
#         # Then cdi = c - cdo
#         term1 = S0*np.exp(-q*T)*norm.cdf(x1)
#         term2 = K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
#         term3 = S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(y1)
#         term4 = K*np.exp(-r*T)*(H/S0)**(2*lam - 2)*norm.cdf(y1 - sigma*np.sqrt(T))
#         cdo   = term1 - term2 - term3 + term4
#         if cdo < 0:
#             cdo = 0
#             cdi   = c - cdo
#             return cdi
#         else:
#             return cdi

#     # --------------------------------
#     # Down-and-out Call
#     # --------------------------------
#     elif option_type == 'down-and-out call' and H <= K:

#         cdi = (S0 * np.exp(-q*T) * (H/S0)**(2*lam) * norm.cdf(y)
#             - K * np.exp(-r*T) * (H/S0)**(2*lam - 2)
#                 * norm.cdf(y - sigma*np.sqrt(T)))
#         cdo = c - cdi
#         if cdo > 0:
#             return cdo
#         else:
#             return 0

#     elif option_type == 'down-and-out call' and H >= K:
#         # This is the “If H > K” formula for the down-and-out call
#         term1 = S0 * np.exp(-q*T)*norm.cdf(x1)
#         term2 = K  * np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
#         term3 = S0 * np.exp(-q*T)*((H/S0)**(2*lam))*norm.cdf(y1)
#         term4 = K  * np.exp(-r*T)*((H/S0)**(2*lam - 2))*norm.cdf(y1 - sigma*np.sqrt(T))
#         cdo   = term1 - term2 - term3 + term4
        
#         if cdo < 0:
#             return 0
#         else:
#             return cdo

#     # --------------------------------
#     # Up-and-in Call
#     # --------------------------------
#     elif option_type == 'up-and-in call' and H > K:
#         # Standard up-and-in call for H > K
#         cui = (S0*np.exp(-q*T)*norm.cdf(x1)
#                - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
#                - S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(-y) - norm.cdf(-y1))
#                + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
#                  * (norm.cdf(-y + sigma*np.sqrt(T))
#                     - norm.cdf(-y1 + sigma*np.sqrt(T))))
#         return cui

#     elif option_type == 'up-and-in call' and H <= K:
#         # If barrier is below K, the up-and-in call is effectively the same as c
#         # or 0, depending on your setup.  Typically if H < S0 < K,
#         # the option knocks in only if S0 goes above H.  If you are sure
#         # you want to treat it as simply c, do so here:
#         return c

#     # --------------------------------
#     # Up-and-out Call
#     # --------------------------------
#     elif option_type == 'up-and-out call' and H <= K:
#         # If the barrier H <= K is below the current spot,
#         # often up-and-out call is worthless if it is truly "up" barrier?
#         return 0.0

#     elif option_type == 'up-and-out call' and H > K:
#         cui = (S0*np.exp(-q*T)*norm.cdf(x1)
#                - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
#                - S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(-y) - norm.cdf(-y1))
#                + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
#                  * (norm.cdf(-y + sigma*np.sqrt(T))
#                     - norm.cdf(-y1 + sigma*np.sqrt(T))))
#         cuo = c - cui
#         return cuo

#     # --------------------------------
#     # Up-and-in Put
#     # --------------------------------
#     elif option_type == 'up-and-in put' and H >= K:
#         pui = (-S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y)
#                + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
#                  * norm.cdf(-y + sigma*np.sqrt(T)))
#         return pui
    
#         # --------------------------------
#     elif option_type == 'up-and-in put' and H <= K:
#         puo = (
#             -S0*np.exp(-q*T)*norm.cdf(-x1)
#             + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
#             + S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y1)
#             - K*np.exp(-r*T)*(H/S0)**(2*lam - 2)*norm.cdf(-y1 + sigma*np.sqrt(T))
#         )
#         if puo < 0:
#             puo = 0
#             pui = black_scholes(S0,K,T,r,sigma,"Put")
#             return pui
#         else:
#             pui = black_scholes(S0,K,T,r,sigma,"Put") - puo
        
#         return pui
    
#     elif option_type == 'up-and-in put' and H <= K:
#         # up-and-in put is the difference p - up-and-out put
#         # but for the simplified logic, we can just return p if the barrier is < K
#         return p

#     # --------------------------------
#     # Up-and-out Put
#     # --------------------------------
#     elif option_type == 'up-and-out put' and H >= K:
#         # puo = p - pui
#         pui = (-S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y)
#                + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
#                  * norm.cdf(-y + sigma*np.sqrt(T)))
#         puo = p - pui
#         return puo

#     elif option_type == 'up-and-out put' and H <= K:
#         # Standard formula for H <= K
#         puo = (
#             -S0*np.exp(-q*T)*norm.cdf(-x1)
#             + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
#             + S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y1)
#             - K*np.exp(-r*T)*(H/S0)**(2*lam - 2)*norm.cdf(-y1 + sigma*np.sqrt(T))
#         )
#         if puo < 0:
#             return 0
#         else:
#             return puo

#     # --------------------------------
#     # Down-and-in Put
#     # --------------------------------
#     elif option_type == 'down-and-in put' and H < K and S0 < H:
#         vanilla = black_scholes(S0, K, T, r, sigma, "Put")
#         return vanilla
    
#     elif option_type == 'down-and-in put' and H > K:
#         # If the barrier is above K, we often treat the down-and-in put as p
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
#         # Typically worthless if H > K in certain setups
#         return 0

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
#         if pdo > 0:
#             return pdo
#         else:
#             return 0

#     # Fallback
#     return None

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
        return cuo

    # --------------------------------
    # Up-and-in Put
    # --------------------------------
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
            return 0
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


# Forward Euler finite difference method
def forward_eulerV(S0, K, T, r, sigma, dS, dt, option_type):
    S_max = 2*max(S0,K)*np.exp(r*T)
    M = int(S_max / dS)
    N = int(T / dt)
    dS = S_max / M
    dt = T / N


    matval = np.zeros((M + 1, N + 1))
    vetS = np.linspace(0, S_max, M + 1)

    # Boundary conditions
    if option_type == "Call":
        matval[:, -1] = np.maximum(vetS - K, 0)
        matval[0, :] = 0
        matval[-1, :] = S_max - K * np.exp(-r * (T - np.linspace(0, T, N + 1)))
    elif option_type == "Put":
        matval[:, -1] = np.maximum(K - vetS, 0)
        matval[0, :] = K * np.exp(-r * (T - np.linspace(0, T, N + 1)))
        matval[-1, :] = 0

    # Coefficients
    a = 0.5 * dt * (sigma**2 * np.arange(M + 1) - r) * np.arange(M + 1)
    b = 1 - dt * (sigma**2 * np.arange(M + 1)**2 + r)
    c = 0.5 * dt * (sigma**2 * np.arange(M + 1) + r) * np.arange(M + 1)

    # Time-stepping
    for j in range(N, 0, -1):
        for i in range(1, M):
            matval[i, j - 1] = (
                a[i] * matval[i - 1, j]
                + b[i] * matval[i, j]
                + c[i] * matval[i + 1, j]
            )

    price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
    price = price_interp(S0)

        
    return price



def backward_eulerV(S0, K, r, T, sigma, dS, dt, option_type):
    # set up grid and adjust increments if necessary
    Smax = 2*max(S0,K)*np.exp(r*T)
    M = round(Smax / dS)
    dS = Smax / M
    N = round(T / dt)
    dt = T / N
    matval = np.zeros((M + 1, N + 1))
    vetS = np.linspace(0, Smax, M + 1)
    veti = np.arange(0, M + 1)
    vetj = np.arange(0, N + 1)
    
    # Boundary conditions
    if option_type == "Call":
        matval[:, -1] = np.maximum(vetS - K, 0)
        matval[0, :] = 0
        #matval[-1, :] = S_max - K * np.exp(-r * (T - np.linspace(0, T, N + 1)))
        matval[-1, :] = Smax - K * np.exp(-r * dt * (N - vetj))
    elif option_type == "Put":
        matval[:, -1] = np.maximum(K - vetS, 0)
        matval[0, :] = K * np.exp(-r * (T - np.linspace(0, T, N + 1)))
        matval[-1, :] = 0
    
    # set up the tridiagonal coefficients matrix
    a = 0.5 * (r * dt * veti - sigma**2 * dt * (veti**2))
    b = 1 + sigma**2 * dt * (veti**2) + r * dt
    c = -0.5 * (r * dt * veti + sigma**2 * dt * (veti**2))
    coeff = np.diag(a[2:M], -1) + np.diag(b[1:M]) + np.diag(c[1:M-1], 1)
    #lu, piv = lu_factor(coeff)
    
    if option_type == "Put":
        
        LU, piv = lu_factor(coeff)  # Equivalent to MATLAB's [L, U] = lu(coeff)

        # Solve the sequence of linear systems
        aux = np.zeros(M-1)

        for j in range(N-1, -1, -1):  # Reverse loop from N to 1
            aux[0] = -a[1] * matval[0, j]  # Adjust indexing for Python (0-based)
    
            # Solve L(Ux) = b using LU decomposition
            matval[1:M, j] = lu_solve((LU, piv), matval[1:M, j+1] + aux)
         
        price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
        price = price_interp(S0)
        
        return price
    
    elif option_type == "Call":
        LU, piv = lu_factor(coeff)  # Equivalent to MATLAB's [L, U] = lu(coeff)

        # Solve the sequence of linear systems
        aux = np.zeros(M-1)

        for j in range(N-1, -1, -1):  # Reverse loop from N to 1
            aux[M-2] = -c[M-1] * matval[M, j]  # Adjust indexing for Python (0-based)
    
            # Solve L(Ux) = b using LU decomposition
            matval[1:M, j] = lu_solve((LU, piv), matval[1:M, j+1] + aux)
            
        price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
        price = price_interp(S0)

        
        return price
    
def crank_nicolsonV(S0, K, r, T, sigma, dS, dt, option_type):
    Smax = 2*max(S0,K)*np.exp(r*T)
    M = round(Smax / dS)
    dS = Smax / M
    N = round(T / dt)
    dt = T / N
    matval = np.zeros((M+1, N+1))
    vetS = np.linspace(0, Smax, M+1)
    veti = np.arange(0, M+1)
    vetj = np.arange(0, N+1)

    if option_type == "Call":
        matval[:, -1] = np.maximum(vetS - K, 0)
        matval[0, :] = 0
        #matval[-1, :] = S_max - K * np.exp(-r * (T - np.linspace(0, T, N + 1)))
        matval[-1, :] = Smax - K * np.exp(-r * dt * (N - vetj))
    elif option_type == "Put":
        matval[:, -1] = np.maximum(K - vetS, 0)
        matval[0, :] = K * np.exp(-r * (T - np.linspace(0, T, N + 1)))
        matval[-1, :] = 0

    # Set up the coefficients matrix
    alpha = 0.25 * dt * (sigma**2 * (veti**2) - r * veti)
    beta = -0.5 * dt * (sigma**2 * (veti**2) + r)
    gamma = 0.25 * dt * (sigma**2 * (veti**2) + r * veti)

    # Construct tridiagonal matrices
    M1 = -np.diag(alpha[2:M], -1) + np.diag(1 - beta[1:M]) - np.diag(gamma[1:M-1], 1)
    M2 = np.diag(alpha[2:M], -1) + np.diag(1 + beta[1:M]) + np.diag(gamma[1:M-1], 1)

    # LU decomposition for efficient solving
    LU, piv = lu_factor(M1)

    # Solve the sequence of linear systems
    lostval = np.zeros(M2.shape[1])

    for j in range(N-1, -1, -1):
        if len(lostval) > 1:
            lostval[0] = alpha[1] * (matval[0, j] + matval[0, j+1])
            lostval[-1] = gamma[-1] * (matval[-1, j] + matval[-1, j+1])
        else:
            lostval = lostval[0] + lostval[-1]

        rhs = M2 @ matval[1:M, j+1] + lostval
        matval[1:M, j] = lu_solve((LU, piv), rhs) 
    
    price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
    price = price_interp(S0)
                
    return price

# # Black-Scholes formula for analytical solution
# def black_scholes(S, K, T, r, sigma, option_type):
#     d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)

#     if option_type == "Call":
#         price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
#         return price
#     elif option_type == "Put":
#         price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
#         return price

# Forward Euler finite difference method
def forward_euler(S0, K, T, r, sigma, dS, dt, option_type, barrier):
    if option_type == "down-and-out put":
        matval, veti, M, N, vetS, r =DOPut(S0,K,r,T,barrier,dS,dt)
    elif option_type == "up-and-out put":
        matval, veti, M, N, vetS, r =UOPut(S0,K,r,T,barrier,dS,dt)
    elif option_type == "up-and-out call":
        matval, veti, M, N, vetS, r =UOCall(S0,K,r,T,barrier,dS,dt)
    elif option_type == "down-and-out call":
        matval, veti, M, N, vetS, r =DOCall(S0,K,r,T,barrier,dS,dt)
    elif option_type == "down-and-in call":
        doc, _, _ = forward_euler(S0, K, T, r, sigma, dS, dt, "down-and-out call", barrier)
        if doc < 0:
            doc = 0
            #price = black_scholes(S0, K, T, r, sigma, "Call") - doc
            price = forward_eulerV(S0, K, T, r, sigma, dS, dt, "Call") - doc
        else:
            #price = black_scholes(S0, K, T, r, sigma, "Call") - doc
            price = forward_eulerV(S0, K, T, r, sigma, dS, dt, "Call") - doc
        return price,_,_
    elif option_type == "down-and-in put":
        dop, _, _ = forward_euler(S0, K, T, r, sigma, dS, dt, "down-and-out put", barrier)
        price = forward_eulerV(S0, K, T, r, sigma, dS, dt, "Put") - dop
        return price,_,_
    elif option_type == "up-and-in put":
        uop, _, _ = forward_euler(S0, K, T, r, sigma, dS, dt, "up-and-out put", barrier)
        if uop < 0:
            uop = 0
            price = forward_eulerV(S0, K, T, r, sigma, dS, dt, "Put") - uop
        else: 
            price = forward_eulerV(S0, K, T, r, sigma, dS, dt, "Put") - uop
        return price,_,_
    elif option_type == "up-and-in call":
        uop, _, _ = forward_euler(S0, K, T, r, sigma, dS, dt, "up-and-out call", barrier)
        price = forward_eulerV(S0, K, T, r, sigma, dS, dt, "Call") - uop
        return price,_,_
    elif option_type not in ("down-and-out put", "up-and-out put", "up-and-out call", "down-and-out call"):
        st.error("UnboundLocalError: The variable 'veti' has not been assigned a value. This is likely due to an unsupported option type. Please select a valid option type, such as 'down-and-out put', 'up-and-out put', 'up-and-out call' or 'down-and-out call'.")
        st.stop()
    else:
        pass
        
    # Coefficients
    a = 0.5 * dt * (((sigma**2) * veti) - r) * veti
    b = 1 - dt * (((sigma**2) * (veti**2)) + r)
    c = 0.5 * dt * (((sigma**2) * veti) + r) * veti

    # Time-stepping
    for j in range(N, 0, -1):
        for i in range(1, M):
            matval[i, j - 1] = (
                a[i] * matval[i - 1, j]
                + b[i] * matval[i, j]
                + c[i] * matval[i + 1, j]
            )

    price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
    price = price_interp(S0)

        
    return price, vetS, matval[:, 0]



def backward_euler(S0, K, r, T, sigma, dS, dt, option_type, barrier):
    if option_type == "down-and-out put":
        matval, veti, M, N, vetS, r =DOPut(S0,K,r,T,barrier,dS,dt)
    elif option_type == "up-and-out put":
        matval, veti, M, N, vetS, r =UOPut(S0,K,r,T,barrier,dS,dt)
    elif option_type == "up-and-out call":
        matval, veti, M, N, vetS, r =UOCall(S0,K,r,T,barrier,dS,dt)
    elif option_type == "down-and-out call":
        matval, veti, M, N, vetS, r =DOCall(S0,K,r,T,barrier,dS,dt)
    elif option_type == "down-and-in call":
        doc, _, _ = backward_euler(S0, K, r, T, sigma, dS, dt, "down-and-out call", barrier)
        if doc < 0:
            doc = 0
            price = backward_eulerV(S0, K, r, T, sigma, dS, dt, "Call") - doc
        else:
            price = backward_eulerV(S0, K, r, T, sigma, dS, dt, "Call") - doc
        return price,_,_
    elif option_type == "down-and-in put":
        dop, _, _ = backward_euler(S0, K, r, T, sigma, dS, dt, "down-and-out put", barrier)
        price = backward_eulerV(S0, K, r, T, sigma, dS, dt, "Put") - dop
        return price,_,_
    elif option_type == "up-and-in put":
        uop, _, _ = backward_euler(S0, K, r, T, sigma, dS, dt, "up-and-out put", barrier)
        if uop < 0:
            uop = 0
            price = backward_eulerV(S0, K, r, T, sigma, dS, dt, "Put") - uop
        else:
            price = backward_eulerV(S0, K, r, T, sigma, dS, dt, "Put") - uop 
        return price,_,_
    elif option_type == "up-and-in call":
        uop, _, _ = backward_euler(S0, K, r, T, sigma, dS, dt, "up-and-out call", barrier)
        price = backward_eulerV(S0, K, r, T, sigma, dS, dt, "Call") - uop
        return price,_,_
    elif option_type not in ("down-and-out put", "up-and-out put", "up-and-out call", "down-and-out call"):
        st.error("UnboundLocalError: The variable 'veti' has not been assigned a value. This is likely due to an unsupported option type. Please select a valid option type, such as 'down-and-out put', 'up-and-out put', 'up-and-out call' or 'down-and-out call'.")
        st.stop()     
    
    # set up the tridiagonal coefficients matrix
    a = 0.5 * (r * dt * veti - sigma**2 * dt * (veti**2))
    b = 1 + sigma**2 * dt * (veti**2) + r * dt
    c = -0.5 * (r * dt * veti + sigma**2 * dt * (veti**2))
    coeff = np.diag(a[2:M], -1) + np.diag(b[1:M]) + np.diag(c[1:M-1], 1)
    #lu, piv = lu_factor(coeff)
    
    if "put" in option_type:
        
        LU, piv = lu_factor(coeff)  # Equivalent to MATLAB's [L, U] = lu(coeff)

        # Solve the sequence of linear systems
        aux = np.zeros(M-1)

        for j in range(N-1, -1, -1):  # Reverse loop from N to 1
            aux[0] = -a[1] * matval[0, j]  # Adjust indexing for Python (0-based)
    
            # Solve L(Ux) = b using LU decomposition
            matval[1:M, j] = lu_solve((LU, piv), matval[1:M, j+1] + aux)
         
        price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
        price = price_interp(S0)
        
        return price, vetS, matval[:, 0]
    
    elif "call" in option_type:
        LU, piv = lu_factor(coeff)  # Equivalent to MATLAB's [L, U] = lu(coeff)

        # Solve the sequence of linear systems
        aux = np.zeros(M-1)

        for j in range(N-1, -1, -1):  # Reverse loop from N to 1
            aux[M-2] = -c[M-1] * matval[M, j]  # Adjust indexing for Python (0-based)
    
            # Solve L(Ux) = b using LU decomposition
            matval[1:M, j] = lu_solve((LU, piv), matval[1:M, j+1] + aux)
            
        price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
        price = price_interp(S0)

        
        return price, vetS, matval[:, 0]
    
def crank_nicolson(S0, K, r, T, sigma, dS, dt, option_type, barrier):

    if option_type == "down-and-out put":
        matval, veti, M, N, vetS, r =DOPut(S0,K,r,T,barrier,dS,dt)
    elif option_type == "up-and-out put":
        matval, veti, M, N, vetS, r =UOPut(S0,K,r,T,barrier,dS,dt)
    elif option_type == "up-and-out call":
        matval, veti, M, N, vetS, r =UOCall(S0,K,r,T,barrier,dS,dt)
    elif option_type == "down-and-out call":
        matval, veti, M, N, vetS, r =DOCall(S0,K,r,T,barrier,dS,dt)
    elif option_type == "down-and-in call":
        doc, _, _ = crank_nicolson(S0, K, r, T, sigma, dS, dt, "down-and-out call", barrier)
        if doc < 0:
            doc = 0
            price = crank_nicolsonV(S0, K, r, T, sigma, dS, dt, "Call") - doc
        else:
            price = crank_nicolsonV(S0, K, r, T, sigma, dS, dt, "Call") - doc
        price = crank_nicolsonV(S0, K, r, T, sigma, dS, dt, "Call") - doc
        return price,_,_
    elif option_type == "down-and-in put":
        dop, _, _ = crank_nicolson(S0, K, r, T, sigma, dS, dt, "down-and-out put", barrier)
        price = crank_nicolsonV(S0, K, r, T, sigma, dS, dt, "Put") - dop
        return price,_,_
    elif option_type == "up-and-in put":
        uop, _, _ = crank_nicolson(S0, K, r, T, sigma, dS, dt, "up-and-out put", barrier)
        if uop < 0:
            uop = 0
            price = crank_nicolsonV(S0, K, r, T, sigma, dS, dt, "Put") - uop
        else: 
            price = crank_nicolsonV(S0, K, r, T, sigma, dS, dt, "Put") - uop
        return price,_,_
    elif option_type == "up-and-in call":
        uop, _, _ = crank_nicolson(S0, K, r, T, sigma, dS, dt, "up-and-out call", barrier)
        price = crank_nicolsonV(S0, K, r, T, sigma, dS, dt, "Call") - uop
        return price,_,_
    elif option_type not in ("down-and-out put", "up-and-out put", "up-and-out call", "down-and-out call"):
        st.error("UnboundLocalError: The variable 'veti' has not been assigned a value. This is likely due to an unsupported option type. Please select a valid option type, such as 'down-and-out put', 'up-and-out put', 'up-and-out call' or 'down-and-out call'.")
        st.stop()        
        
    # Set up the coefficients matrix
    alpha = 0.25 * dt * (sigma**2 * (veti**2) - r * veti)
    beta = -0.5 * dt * (sigma**2 * (veti**2) + r)
    gamma = 0.25 * dt * (sigma**2 * (veti**2) + r * veti)

    # Construct tridiagonal matrices
    M1 = -np.diag(alpha[2:M], -1) + np.diag(1 - beta[1:M]) - np.diag(gamma[1:M-1], 1)
    M2 = np.diag(alpha[2:M], -1) + np.diag(1 + beta[1:M]) + np.diag(gamma[1:M-1], 1)

    # LU decomposition for efficient solving
    LU, piv = lu_factor(M1)

    # Solve the sequence of linear systems
    lostval = np.zeros(M2.shape[1])

    for j in range(N-1, -1, -1):
        if len(lostval) > 1:
            lostval[0] = alpha[1] * (matval[0, j] + matval[0, j+1])
            lostval[-1] = gamma[-1] * (matval[-1, j] + matval[-1, j+1])
        else:
            lostval = lostval[0] + lostval[-1]

        rhs = M2 @ matval[1:M, j+1] + lostval
        matval[1:M, j] = lu_solve((LU, piv), rhs) 
    
    price_interp = interp1d(vetS, matval[:, 0], kind='linear', fill_value="extrapolate")
    price = price_interp(S0)
                
    return price, vetS, matval[:, 0]

# def crank_nicolson_in(S0, K, r, T, sigma, dS, dt, option_type, barrier):

#     if option_type == "down-and-out put":
#         matval, veti, M, N, vetS, r =DOPut(S0,K,r,T,barrier,dS,dt)
#     elif option_type == "up-and-out put":
#         matval, veti, M, N, vetS, r =UOPut(S0,K,r,T,barrier,dS,dt)
#     elif option_type == "up-and-out call":
#         matval, veti, M, N, vetS, r =UOCall(S0,K,r,T,barrier,dS,dt)
#     elif option_type == "down-and-out call":
#         matval, veti, M, N, vetS, r =DOCall(S0,K,r,T,barrier,dS,dt)
#     elif option_type == "down-and-in call":
#         doc, _, _ = crank_nicolson(S0, K, T, r, sigma, dS, dt, "down-and-out call", barrier)
#         price = black_scholes(S0, K, T, r, sigma, "Call") - doc
#         return price,_,_
#     elif option_type == "down-and-in put":
#         dop, _, _ = crank_nicolson(S0, K, T, r, sigma, dS, dt, "down-and-out put", barrier)
#         price = black_scholes(S0, K, T, r, sigma, "Call") - dop
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

#     return price

def DOPut(S0,K,r,T,barrier,dS,dt):
    Smax = 2*max(S0,K)*np.exp(r*T)
    M = round((Smax-barrier)/dS)
    dS = (Smax-barrier)/M
    N = round(T/dt)
    dt = T/N
    matval = np.zeros((M+1,N+1))
    vetS = np.linspace(barrier,Smax,M+1)
    veti = vetS/dS
    vetj = np.arange(0, N+1)
    
    matval[:,-1] = np.maximum(K-vetS,0)
    matval[0,:] = 0
    matval[-1,:] = 0
    
    return matval, veti, M, N, vetS, r

def UOPut(S0,K,r,T,barrier,dS,dt):
    Smax = 2*max(S0,K)*np.exp(r*T)
    M = round((barrier)/dS)
    dS = (barrier)/M
    N = round(T/dt)
    dt = T/N
    matval = np.zeros((M+1,N+1))
    vetS = np.linspace(0,barrier,M+1)
    veti = vetS/dS
    vetj = np.arange(0, N+1)

    matval[:,-1] = np.maximum(K-vetS,0)
    matval[0,:] = 0
    matval[-1,:] = 0
    
    return matval, veti, M, N, vetS, r
    
def UOCall(S0,K,r,T,barrier,dS,dt):
    Smax = 2*max(S0,K)*np.exp(r*T)
    M = round((barrier)/dS)
    dS = (barrier)/M
    N = round(T/dt)
    dt = T/N
    matval = np.zeros((M+1,N+1))
    vetS = np.linspace(0,barrier,M+1)
    veti = vetS/dS
    vetj = np.arange(0, N+1)

    matval[:,-1] = np.maximum(vetS-K,0)
    matval[0,:] = 0
    matval[-1,:] = 0
    
    return matval, veti, M, N, vetS, r

def DOCall(S0,K,r,T,barrier,dS,dt):
    Smax = 2*max(S0,K)*np.exp(r*T)
    M = round((Smax-barrier)/dS)
    dS = (Smax-barrier)/M
    N = round(T/dt)
    dt = T/N
    matval = np.zeros((M+1,N+1))
    vetS = np.linspace(barrier,Smax,M+1)
    veti = vetS/dS
    vetj = np.arange(0, N+1)
    
    matval[:,-1] = np.maximum(vetS-K,0)
    matval[0,:] = 0 
    matval[-1,:] = Smax-K*np.exp(-r * dt*(N-vetj))
    
    return matval, veti, M, N, vetS, r

# Streamlit interface
st.title("Comparison of different numerical schemes and the analytical solution for barrier options")

 #S_max = st.sidebar.number_input("Maximum Stock Price (S_max)", value=200.0, step=1.0)
S0 = st.sidebar.number_input("Spot Price (S0)", value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0, step=0.1)
r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05, step=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01)
dS = st.sidebar.number_input("Stock Price Step (dS)", value=10.0, step=0.1)
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
    ]
)
barrier = st.sidebar.number_input("Barrier", value=80.0, step=0.01)
q = st.sidebar.number_input("Dividend Yield (q)", value=0.03)
rebate = st.sidebar.number_input("Rebate", value=0.0)
numerical_method = st.sidebar.selectbox("Numerical method", ("Forward Euler", "Backward Euler", "Crank-Nicolson"))
     

# if numerical_method == "Forward Euler":
#     # Compute Forward Euler results
#     price, S_grid, forward_euler_prices = forward_euler(S0, K, T, r, sigma, dS, dt, option_type, barrier)
#     if price < 0:
#         price = 0.0
        
        

#     # Compute the analytical barrier option price at the spot price S0.
#     analytical_price = barrier_option_price(S0, K, T, r, q, sigma, barrier, option_type)
#     # Compute analytical prices over the entire grid S_grid.
#     #analytical_prices =barrier_option_price(S_grid, K, T, r, q, sigma, barrier, option_type)
    
#     forward_euler_prices_1d = []
#     for i in range(len(forward_euler_prices)):
#         price_interp = interp1d(S_grid, forward_euler_prices, kind='linear', fill_value="extrapolate")
#         price = price_interp()
#         forward_euler_prices_1d.append(price)
                                  
#     # Compute the absolute errors.
#     #absolute_error = np.abs(np.array(forward_euler_prices) - analytical_prices)

#     # Create a DataFrame for the comparison at S0.
#     df = pd.DataFrame({
#         "Forward Euler Price": [np.round(price, 4)],
#         "Analytical Price": [np.round(analytical_price, 4)],
#         "Absolute Error": [np.round(np.abs(price - analytical_price), 4)]
#     })

#     st.subheader("Option Price Comparison at Spot Price (S0)")
#     st.table(df)

#     # Plot the numerical and analytical prices over the grid.
#     st.subheader("Comparison of Prices Across All Stock Prices")
#     fig = go.Figure()

#     # Scatter plot for Forward Euler results.
#     fig.add_trace(go.Scatter(
#         x=S_grid,
#         y=forward_euler_prices,
#         mode="markers",
#         name="Forward Euler Prices",
#         marker=dict(color="red", size=6)
#     ))

#     #Line plot for Analytical Barrier Option Prices.
#     # fig.add_trace(go.Scatter(
#     #     x=S_grid,
#     #     y=analytical_prices,
#     #     mode="lines",
#     #     name="Analytical Barrier Prices",
#     #     line=dict(color="blue", width=2)
#     # ))

#     fig.update_layout(
#         title="Option Prices: Forward Euler vs Analytical",
#         xaxis_title="Stock Price (S)",
#         yaxis_title="Option Price (V)",
#         legend_title="Method",
#         width=800,
#         height=500
#     )

#     st.plotly_chart(fig)

if numerical_method == "Forward Euler":
    # Compute Forward Euler results
    num_price, S_grid, forward_euler_prices = forward_euler(S0, K, T, r, sigma, dS, dt, option_type, barrier)
    if num_price < 0:
        num_price = 0.0

    # Compute the analytical barrier option price at S0.
    analytical_price = barrier_option_price(S0, K, T, r, q, sigma, barrier, option_type)
    
    # Create an interpolation function from the numerical scheme results.
    price_interp = interp1d(S_grid, forward_euler_prices, kind='linear', fill_value="extrapolate")
    # Evaluate the interpolation on the same grid (or you could use a new grid if desired)
    forward_euler_prices_1d = price_interp(S_grid)
                                  
    # Create a DataFrame for the comparison at S0.
    df = pd.DataFrame({
        "Forward Euler Price": [np.round(num_price, 4)],
        "Analytical Price": [np.round(analytical_price, 4)],
        "Absolute Error": [np.round(np.abs(num_price - analytical_price), 4)]
    })
    st.subheader("Option Price Comparison at Spot Price (S0)")
    st.table(df)
    
    # Create a full grid for analytical prices over a broader range.
    S_grid_full = np.linspace(S_grid[0], S_grid[-1], 100)
    analytical_prices = np.array([barrier_option_price(S, K, T, r, q, sigma, barrier, option_type)
                                  for S in S_grid_full])
    
    # Plot the numerical (interpolated) and analytical prices.
    st.subheader("Comparison of Prices Across All Stock Prices")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=S_grid,
        y=forward_euler_prices_1d,
        mode="markers",
        name="Forward Euler Prices",
        marker=dict(color="red", size=6)
    ))
    fig.add_trace(go.Scatter(
        x=S_grid_full,
        y=analytical_prices,
        mode="lines",
        name="Analytical Barrier Prices",
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
    price, S_grid, backward_euler_prices = backward_euler(S0, K, r, T, sigma, dS, dt, option_type, barrier)
    
    if price < 0:
        price = 0.0
    else:
        pass

    # Compute analytical Black-Scholes prices
    analytical_price = barrier_option_price(S0, K, T, r, q, sigma, barrier, option_type)
    # Compute analytical prices over the entire grid S_grid.
    analytical_prices =barrier_option_price(S_grid, K, T, r, q, sigma, barrier, option_type)
                                  

    # Compute the absolute errors.
    absolute_error = np.abs(np.array(backward_euler_prices) - analytical_prices)


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
    price, S_grid, crank_nicolson_prices= crank_nicolson(S0, K, r, T, sigma, dS, dt, option_type, barrier)
    
    if price < 0:
        price = 0.0
    else:
        pass

    # Compute analytical Black-Scholes prices
    #analytical_price = black_scholes(S0, K, T, r, sigma, option_type)
    #analytical_prices = black_scholes(S_grid, K, T, r, sigma, option_type)
    # Find the index closest to S0
    index_S0 = (np.abs(S_grid - S0)).argmin()


    # Create a DataFrame for comparison at S0
    df = pd.DataFrame({
        "Crank Nicolson Price": [np.round(price, 4)],
        #"Analytical Price": [analytical_price],
        #"Absolute Error": [np.abs(price - analytical_price)],
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
    # fig.add_trace(go.Scatter(
    #     x=S_grid, 
    #     y=analytical_prices, 
    #     mode="lines", 
    #     name="Analytical Black-Scholes Prices",
    #     line=dict(color="blue", width=2)
    # ))

    fig.update_layout(
        title="Option Prices: Crank Nicolson vs Analytical",
        xaxis_title="Stock Price (S)",
        yaxis_title="Option Price (V)",
        legend_title="Method",
        width=800,
        height=500
    )
    st.plotly_chart(fig)





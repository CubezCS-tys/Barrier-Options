
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

def crank_nicolson_vanilla_call(S0, K, T, r, sigma, dS, dt):
    """
    Crank–Nicolson PDE for a vanilla European call on [0, S_max].
    Returns: (price, S_grid, V_at_t0).
    """
    eps = 1e-12  # define the error threshold
    S_max = 2 * max(S0, K) * np.exp(r * T)
    M = int(S_max / dS)
    N = int(T / dt)
    dS = S_max / M  # adjust
    dt = T / N      # adjust

    S_grid = np.linspace(0, S_max, M + 1)
    V = np.zeros((N + 1, M + 1))
    # Terminal payoff:
    V[-1, :] = np.maximum(S_grid - K, 0.0)
    
    # Precompute Crank–Nicolson coefficients for j = 0,...,M.
    # Here we define:
    #   a[j] = 0.25 * dt * (sigma**2 * j**2 - r * j)
    #   b[j] = 0.5  * dt * (sigma**2 * j**2 + r)
    #   c[j] = 0.25 * dt * (sigma**2 * j**2 + r * j)
    j_arr = np.arange(M + 1)
    a = 0.25 * dt * (sigma**2 * j_arr**2 - r * j_arr)
    b = 0.5  * dt * (sigma**2 * j_arr**2 + r)
    c = 0.25 * dt * (sigma**2 * j_arr**2 + r * j_arr)
    
    # Build the tridiagonal matrix for interior nodes j = 1,...,M-1.
    main_diag = 1 + b[1:M]
    lower_diag = -a[2:M]      # corresponds to V_{j-1}^{n-1} for j = 2,...,M-1
    upper_diag = -c[1:M-1]    # corresponds to V_{j+1}^{n-1} for j = 1,...,M-2
    LHS = np.diag(main_diag)
    if M - 2 > 0:
        LHS += np.diag(lower_diag, k=-1) + np.diag(upper_diag, k=1)
    
    # Time-stepping (backward in time)
    t_arr = np.linspace(0, T, N + 1)
    for n in range(N, 0, -1):
        tau = T - t_arr[n - 1]
        # Boundary conditions for a call:
        V[n - 1, 0]   = 0.0
        V[n - 1, -1]  = S_max - K * np.exp(-r * tau)
        
        # Build right-hand side for interior nodes j = 1,...,M-1.
        # Using the explicit part:
        #   rhs[j-1] = a[j]*V[n, j-1] + (1 - b[j])*V[n, j] + c[j]*V[n, j+1]
        rhs = a[1:M] * V[n, 0:M-1] + (1 - b[1:M]) * V[n, 1:M] + c[1:M] * V[n, 2:M+1]
        # Adjust for known boundary values:
        # For j = 1 (leftmost interior): add a[1]*V[n-1,0] (V[n-1,0] is already set)
        rhs[0]   += a[1] * V[n - 1, 0]
        # For j = M-1: add c[M-1]*V[n-1,-1]
        rhs[-1]  += c[M - 1] * V[n - 1, -1]
        
        # Solve for interior nodes:
        V[n - 1, 1:M] = np.linalg.solve(LHS, rhs)
        V[n - 1, :] = np.where(np.abs(V[n - 1, :]) < eps, 0.0, V[n - 1, :])
    
    # Interpolate to get the price at S0:
    interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
    price = float(interp_fn(S0))
    return price, S_grid, V[0, :]


def crank_nicolson_vanilla_put(S0, K, T, r, sigma, dS, dt):
    """
    Crank–Nicolson PDE for a vanilla European put on [0, S_max].
    Returns: (price, S_grid, V_at_t0).
    """
    eps = 1e-12  # define the error threshold
    S_max = 2 * max(S0, K) * np.exp(r * T)
    M = int(S_max / dS)
    N = int(T / dt)
    dS = S_max / M
    dt = T / N

    S_grid = np.linspace(0, S_max, M + 1)
    V = np.zeros((N + 1, M + 1))
    # Terminal payoff:
    V[-1, :] = np.maximum(K - S_grid, 0.0)
    
    j_arr = np.arange(M + 1)
    a = 0.25 * dt * (sigma**2 * j_arr**2 - r * j_arr)
    b = 0.5  * dt * (sigma**2 * j_arr**2 + r)
    c = 0.25 * dt * (sigma**2 * j_arr**2 + r * j_arr)
    
    main_diag = 1 + b[1:M]
    lower_diag = -a[2:M]
    upper_diag = -c[1:M-1]
    LHS = np.diag(main_diag)
    if M - 2 > 0:
        LHS += np.diag(lower_diag, k=-1) + np.diag(upper_diag, k=1)
    
    t_arr = np.linspace(0, T, N + 1)
    for n in range(N, 0, -1):
        tau = T - t_arr[n - 1]
        # Boundary conditions for a put:
        V[n - 1, 0]   = K * np.exp(-r * tau)
        V[n - 1, -1]  = 0.0
        
        rhs = a[1:M] * V[n, 0:M-1] + (1 - b[1:M]) * V[n, 1:M] + c[1:M] * V[n, 2:M+1]
        rhs[0]   += a[1] * V[n - 1, 0]
        rhs[-1]  += c[M - 1] * V[n - 1, -1]
        
        V[n - 1, 1:M] = np.linalg.solve(LHS, rhs)
        V[n - 1, :] = np.where(np.abs(V[n - 1, :]) < eps, 0.0, V[n - 1, :])
    
    interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
    price = float(interp_fn(S0))
    return price, S_grid, V[0, :]


###############################################################################
# 2) Barrier Option Pricing using Crank–Nicolson
###############################################################################
def crank_nicolson_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type):
    """
    Crank–Nicolson for a knock–out call (either 'down-and-out' or 'up-and-out').
    barrier_type: 'down' or 'up'
    
    The terminal payoff is set to zero in the barrier region:
      - For 'down-and-out': zero for S <= barrier.
      - For 'up-and-out':   zero for S >= barrier.
    """
    eps = 1e-12  # define the error threshold
    S_max = 2 * max(S0, K) * np.exp(r * T)
    M = int(S_max / dS)
    N = int(T / dt)
    dS = S_max / M
    dt = T / N

    S_grid = np.linspace(0, S_max, M + 1)
    # Define terminal payoff and apply knockout condition:
    payoff = np.maximum(S_grid - K, 0.0)
    if barrier_type == 'down':
        payoff[S_grid <= barrier] = 0.0
    else:  # 'up'
        payoff[S_grid >= barrier] = 0.0
    V = np.zeros((N + 1, M + 1))
    V[-1, :] = payoff

    j_arr = np.arange(M + 1)
    a = 0.25 * dt * (sigma**2 * j_arr**2 - r * j_arr)
    b = 0.5  * dt * (sigma**2 * j_arr**2 + r)
    c = 0.25 * dt * (sigma**2 * j_arr**2 + r * j_arr)
    
    main_diag = 1 + b[1:M]
    lower_diag = -a[2:M]
    upper_diag = -c[1:M-1]
    LHS = np.diag(main_diag)
    if M - 2 > 0:
        LHS += np.diag(lower_diag, k=-1) + np.diag(upper_diag, k=1)
    
    t_arr = np.linspace(0, T, N + 1)
    for n in range(N, 0, -1):
        tau = T - t_arr[n - 1]
        # Boundary conditions for a call:
        V[n - 1, 0]   = 0.0
        V[n - 1, -1]  = S_max - K * np.exp(-r * tau)
        
        rhs = a[1:M] * V[n, 0:M-1] + (1 - b[1:M]) * V[n, 1:M] + c[1:M] * V[n, 2:M+1]
        rhs[0]   += a[1] * V[n - 1, 0]
        rhs[-1]  += c[M - 1] * V[n - 1, -1]
        
        V[n - 1, 1:M] = np.linalg.solve(LHS, rhs)
        V[n - 1, :] = np.where(np.abs(V[n - 1, :]) < eps, 0.0, V[n - 1, :])
        # Enforce barrier condition at this time level:
        if barrier_type == 'down':
            V[n - 1, S_grid <= barrier] = 0.0
        else:  # 'up'
            V[n - 1, S_grid >= barrier] = 0.0
    
    interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
    price_ko = float(interp_fn(S0))
    return price_ko, S_grid, V[0, :]


def crank_nicolson_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type):
    """
    Crank–Nicolson for a knock–out put (either 'down-and-out' or 'up-and-out').
    barrier_type: 'down' or 'up'
    """
    eps = 1e-12  # define the error threshold
    S_max = 2 * max(S0, K) * np.exp(r * T)
    M = int(S_max / dS)
    N = int(T / dt)
    dS = S_max / M
    dt = T / N

    S_grid = np.linspace(0, S_max, M + 1)
    payoff = np.maximum(K - S_grid, 0.0)
    if barrier_type == 'down':
        payoff[S_grid <= barrier] = 0.0
    else:
        payoff[S_grid >= barrier] = 0.0
    V = np.zeros((N + 1, M + 1))
    V[-1, :] = payoff

    j_arr = np.arange(M + 1)
    a = 0.25 * dt * (sigma**2 * j_arr**2 - r * j_arr)
    b = 0.5  * dt * (sigma**2 * j_arr**2 + r)
    c = 0.25 * dt * (sigma**2 * j_arr**2 + r * j_arr)
    
    main_diag = 1 + b[1:M]
    lower_diag = -a[2:M]
    upper_diag = -c[1:M-1]
    LHS = np.diag(main_diag)
    if M - 2 > 0:
        LHS += np.diag(lower_diag, k=-1) + np.diag(upper_diag, k=1)
    
    t_arr = np.linspace(0, T, N + 1)
    for n in range(N, 0, -1):
        tau = T - t_arr[n - 1]
        # Boundary conditions for a put:
        V[n - 1, 0]   = K * np.exp(-r * tau)
        V[n - 1, -1]  = 0.0
        
        rhs = a[1:M] * V[n, 0:M-1] + (1 - b[1:M]) * V[n, 1:M] + c[1:M] * V[n, 2:M+1]
        rhs[0]   += a[1] * V[n - 1, 0]
        rhs[-1]  += c[M - 1] * V[n - 1, -1]
        
        V[n - 1, 1:M] = np.linalg.solve(LHS, rhs)
        V[n - 1, :] = np.where(np.abs(V[n - 1, :]) < eps, 0.0, V[n - 1, :])
        # Enforce barrier condition:
        if barrier_type == 'down':
            V[n - 1, S_grid <= barrier] = 0.0
        else:
            V[n - 1, S_grid >= barrier] = 0.0
    
    interp_fn = interp1d(S_grid, V[0, :], kind='linear', fill_value='extrapolate')
    price_ko = float(interp_fn(S0))
    return price_ko, S_grid, V[0, :]


###############################################################################
# 3) Main Crank–Nicolson Barrier Wrapper
###############################################################################
def crank_nicolson(S0, K, T, r, sigma, dS, dt, barrier, option_type):
    """
    Main wrapper for Crank–Nicolson pricing of barrier options.
    Knock–in options are obtained via in–out parity:
         knock_in = vanilla - knock_out.
    option_type must be one of:
      "down-and-out call", "down-and-in call", "down-and-out put", "down-and-in put",
      "up-and-out call",   "up-and-in call",   "up-and-out put",   "up-and-in put".
    """
    if option_type == "down-and-out call":
        return crank_nicolson_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down')
    elif option_type == "down-and-in call":
        price_doc, Sg, V_doc = crank_nicolson_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down')
        price_van, Sg, V_van = crank_nicolson_vanilla_call(S0, K, T, r, sigma, dS, dt)
        price_din = price_van - price_doc
        V_din = V_van - V_doc
        return price_din, Sg, V_din
    elif option_type == "down-and-out put":
        return crank_nicolson_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down')
    elif option_type == "down-and-in put":
        price_dop, Sg, V_dop = crank_nicolson_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='down')
        price_van, Sg, V_van = crank_nicolson_vanilla_put(S0, K, T, r, sigma, dS, dt)
        price_din = price_van - price_dop
        V_din = V_van - V_dop
        return price_din, Sg, V_din 
    elif option_type == "up-and-out call":
        return crank_nicolson_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up')
    elif option_type == "up-and-in call":
        price_uoc, Sg, V_uoc = crank_nicolson_knock_out_call(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up')
        price_van, Sg, V_van = crank_nicolson_vanilla_call(S0, K, T, r, sigma, dS, dt)
        price_uic = price_van - price_uoc
        V_uic = V_van - V_uoc
        return price_uic, Sg, V_uic
    elif option_type == "up-and-out put":
        return crank_nicolson_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up')
    elif option_type == "up-and-in put":
        price_uop, Sg, V_uop = crank_nicolson_knock_out_put(S0, K, T, r, sigma, dS, dt, barrier, barrier_type='up')
        price_van, Sg, V_van = crank_nicolson_vanilla_put(S0, K, T, r, sigma, dS, dt)
        price_uip = price_van - price_uop
        V_uip = V_van - V_uop
        return price_uip, Sg, V_uip
    return None

import pandas as pd
import time



import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time  # for timing in convergence tests
from scipy.interpolate import interp1d
from scipy.stats import norm

# -----------------------------------------------------------
# 1) PDE Solvers + Analytical barrier function go here
#    (Make sure these are defined, as usual.)
#    e.g.:
#      - def barrier_option_price(...)
#      - def forward_euler(...)
#      - def backward_euler(...)
#      - def crank_nicolson(...)
# -----------------------------------------------------------

def get_pde_price(method_name, S0, K, T, r, sigma, dS, dt, barrier, option_type):
    """
    Returns the PDE price at spot S0 for the given method_name.
    method_name must be one of: 'FE', 'BE', 'CN'.
    """
    if method_name == 'FE':
        price, _, _ = forward_euler(S0, K, T, r, sigma, dS, dt, barrier, option_type)
    elif method_name == 'BE':
        price, _, _ = backward_euler(S0, K, T, r, sigma, dS, dt, barrier, option_type)
    elif method_name == 'CN':
        price, _, _ = crank_nicolson(S0, K, T, r, sigma, dS, dt, barrier, option_type)
    else:
        price = None
    return 0.0 if (price is None or price < 0) else price


###############################################################################
# Streamlit app
###############################################################################
def app():
    st.set_page_config(page_title="Numerical Scheme Comparisons", layout="wide")
    st.title("Comparison of Forward/Backward/Crank–Nicolson Methods")

    # -------------------------------
    # Sidebar Inputs
    # -------------------------------
    st.sidebar.header("Option & FD Parameters")
    option_type = st.sidebar.selectbox(
        "Option Type", [
            "down-and-in call", "down-and-out call",
            "down-and-in put",  "down-and-out put",
            "up-and-in call",   "up-and-out call",
            "up-and-in put",    "up-and-out put"
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

    # -------------------------------
    # 1) Accuracy Table vs Analytical Value
    # -------------------------------
    st.subheader("1. Accuracy Table vs Analytical Value")

    rows = []
    spots = np.arange(S_min_input, S_max_input + 0.01, S_step)

    # -------------------------------
    # 2) Convergence & Runtime Analysis
    # -------------------------------
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
            
            true_val = barrier_option_price(S0_test, K, T, r, q, sigma, barrier, option_type)
            if true_val is None or true_val < 0:
                true_val = 0.0

            for dt in dt_vals:
                # Forward Euler
                t0 = time.perf_counter()
                fe, _, _ = forward_euler(S0_test, K, T, r, sigma, dS_explicit, dt, barrier, option_type)
                times_FE.append(time.perf_counter() - t0)
                errors_FE.append(abs(fe - true_val))

                # Backward Euler
                t0 = time.perf_counter()
                be, _, _ = backward_euler(S0_test, K, T, r, sigma, dS_implicit, dt, barrier, option_type)
                times_BE.append(time.perf_counter() - t0)
                errors_BE.append(abs(be - true_val))

                # Crank–Nicolson
                t0 = time.perf_counter()
                cn, _, _ = crank_nicolson(S0_test, K, T, r, sigma, dS_CN, dt, barrier, option_type)
                times_CN.append(time.perf_counter() - t0)
                errors_CN.append(abs(cn - true_val))
            
            # Plot Error vs dt (log-log)
            fig_err = go.Figure()
            fig_err.add_trace(go.Scatter(x=dt_vals, y=errors_FE,
                                         mode='lines+markers', name="Forward Euler"))
            fig_err.add_trace(go.Scatter(x=dt_vals, y=errors_BE,
                                         mode='lines+markers', name="Backward Euler"))
            fig_err.add_trace(go.Scatter(x=dt_vals, y=errors_CN,
                                         mode='lines+markers', name="Crank–Nicolson"))
            fig_err.update_layout(title=f"Error vs dt (S₀ = {S0_test})",
                                  xaxis_title="dt",
                                  yaxis_title="Absolute Error",
                                  xaxis_type="log",
                                  yaxis_type="log",
                                  height=500)
            st.plotly_chart(fig_err, use_container_width=True)
            
            # Plot Runtime vs dt
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(x=dt_vals, y=times_FE,
                                          mode='lines+markers', name="Forward Euler"))
            fig_time.add_trace(go.Scatter(x=dt_vals, y=times_BE,
                                          mode='lines+markers', name="Backward Euler"))
            fig_time.add_trace(go.Scatter(x=dt_vals, y=times_CN,
                                          mode='lines+markers', name="Crank–Nicolson"))
            fig_time.update_layout(title=f"Runtime vs dt (S₀ = {S0_test})",
                                   xaxis_title="dt",
                                   yaxis_title="CPU Time (s)",
                                   xaxis_type="log",
                                   height=500)
            st.plotly_chart(fig_time, use_container_width=True)
    
    

if __name__ == "__main__":
    app()

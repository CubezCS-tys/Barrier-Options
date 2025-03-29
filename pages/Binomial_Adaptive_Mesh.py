# import streamlit as st
# import math

# def partial_refined_binomial_down_and_out_call(
#     S0, K, r, sigma, T,
#     N,          # total time steps
#     barrier,    # down-and-out barrier
#     refine_band,# how many "levels" above the barrier to refine
#     rebate=0.0
# ):
#     """
#     Prices a down-and-out call using a partial refinement near the barrier:
#     - Each time step is discrete.
#     - Away from the barrier, we use 1 up and 1 down factor (like CRR).
#     - If the node is within refine_band steps (in price terms) above the barrier,
#       we refine further (2 sub-ups, 2 sub-downs) to get finer resolution near H.
    
#     This is a conceptual approach and not guaranteed arbitrage-free or perfectly recombining.
#     """

#     dt = T / N
#     disc = math.exp(-r * dt)

#     # Base up/down for standard region
#     u_std = math.exp(sigma * math.sqrt(dt))
#     d_std = 1.0 / u_std
    
#     # A smaller factor for the refined region
#     # e.g. each "up" is split into 2 smaller ups: up1 * up2 ~ u_std
#     # Here, for simplicity, let's define them to multiply to the same overall factor:
#     u1 = math.sqrt(u_std)  # so that u1 * u1 = u_std
#     d1 = 1.0 / u1

#     # Each node will be stored as a dict: { "S": current stock price, "V": option value }
#     # We'll keep them in a list at each time step.
#     nodes = [[{"S": S0, "V": None}]]  # time 0 has 1 node

#     # Build forward the price tree with partial refinement
#     for i in range(1, N+1):
#         prev_nodes = nodes[i-1]
#         new_layer = []
#         # We'll track visited prices in a dict to unify recombining nodes
#         price_map = {}
        
#         for nd in prev_nodes:
#             S_prev = nd["S"]
#             # if S_prev <= barrier => knocked out => won't branch
#             if S_prev <= barrier:
#                 continue

#             # Decide if we are near the barrier
#             # Let's measure the "distance" in multiples of (S0 - barrier)/N or something simpler
#             # For example, refine_band steps => we interpret each "price step" as (S0 - barrier)/N
#             step_price = (S0 - barrier) / float(N)
#             dist_from_barrier = S_prev - barrier
#             near_barrier = (dist_from_barrier <= refine_band * step_price)

#             if near_barrier:
#                 # Use refined transitions: 2 up, 2 down
#                 # We'll do up-up, up-down, down-up, down-down, for instance
#                 # But let's simplify and do 2 transitions: "super up" (u1^2 = u_std) and "super down" (d1^2 = d_std)
                
#                 # Actually let's do "2 sub-ups, 2 sub-downs" in total => 4 children, to illustrate
#                 # Child 1: up1 from S_prev
#                 # Child 2: up2 from S_child1
#                 # etc.
                
#                 # We'll just define them manually:
#                 #   S_up_up   = S_prev * (u1 * u1) = S_prev * u_std
#                 #   S_up_down = S_prev * (u1 * d1) = ~ S_prev
#                 #   S_down_up = S_prev * (d1 * u1) = ~ S_prev
#                 #   S_down_down= S_prev*(d1*d1)=S_prev*d_std
#                 # But up_down and down_up are nearly the same -> they recombine => we can unify them
                
#                 # For demonstration, let's do just 2 children: up_up and down_down
#                 S_up = S_prev * u_std  # up-up
#                 S_dn = S_prev * d_std  # down-down
#                 for S_new in [S_up, S_dn]:
#                     if S_new <= barrier:
#                         # node is knocked out
#                         payoff = rebate
#                     else:
#                         payoff = None  # unknown yet
#                     # store in new_layer
#                     if S_new not in price_map:
#                         price_map[S_new] = {"S": S_new, "V": payoff}
#                     else:
#                         # node might exist => no immediate recombination if we're being naive
#                         # but let's unify if S_new matches exactly
#                         # The payoff is not known, so we keep None or rebate if knocked out
#                         if price_map[S_new]["V"] is None:
#                             price_map[S_new]["V"] = payoff
#             else:
#                 # normal branching
#                 S_up = S_prev * u_std
#                 S_dn = S_prev * d_std
#                 for S_new in [S_up, S_dn]:
#                     if S_new <= barrier:
#                         payoff = rebate
#                     else:
#                         payoff = None
#                     if S_new not in price_map:
#                         price_map[S_new] = {"S": S_new, "V": payoff}
#                     else:
#                         if price_map[S_new]["V"] is None:
#                             price_map[S_new]["V"] = payoff
        
#         new_layer = list(price_map.values())
#         nodes.append(new_layer)

#     # Now we do backward induction of option values
#     # We'll go from t=N down to t=0.
#     # At t=N, we can define the payoff if not knocked out
#     # i.e. if node["V"] is None, compute payoff
#     for nd in nodes[N]:
#         if nd["V"] is None:  # not knocked out
#             # payoff for a call
#             nd["V"] = max(nd["S"] - K, 0)

#     # We'll step backwards
#     for i in range(N, 0, -1):
#         layer = nodes[i]
#         prev_layer = nodes[i-1]
#         dt = T / N
#         disc = math.exp(-r * dt)
        
#         # For a simple approach, let's define p from CRR (but we used the same u_std, d_std):
#         u_std = math.exp(sigma * math.sqrt(dt))
#         d_std = 1/u_std
#         p = (math.exp(r * dt) - d_std) / (u_std - d_std)
        
#         # We'll find for each node in prev_layer, which 2 children it leads to in layer
#         # Then V_prev = e^{-r dt} [p V_up + (1-p) V_dn]
        
#         # we need a function to find children states given S_prev
#         def get_children_states(S_prev, i, barrier, refine_band, step_price):
#             dist = S_prev - barrier
#             if dist <= 0:
#                 return []
#             near = (dist <= refine_band * step_price)
#             if near:
#                 # refined => 2 children: S_prev*u_std, S_prev*d_std
#                 return [S_prev * u_std, S_prev * d_std]
#             else:
#                 # normal => same 2 children
#                 return [S_prev * u_std, S_prev * d_std]
        
#         # build a map from price to index in layer
#         price_to_index = {}
#         for idx, node in enumerate(layer):
#             price_to_index[node["S"]] = idx
        
#         # update values in prev_layer
#         for nd_prev in prev_layer:
#             S_prev = nd_prev["S"]
#             # If knocked out => do nothing
#             if nd_prev["S"] <= barrier:
#                 nd_prev["V"] = rebate
#                 continue
            
#             # find children
#             step_price = (S0 - barrier)/float(N)
#             child_prices = get_children_states(S_prev, i, barrier, refine_band, step_price)
#             if len(child_prices) == 0:
#                 # no children => knocked out or trivial
#                 nd_prev["V"] = rebate
#                 continue
#             # compute weighted payoff
#             Vs = []
#             for cpr in child_prices:
#                 # find the node in layer
#                 child_idx = price_to_index.get(cpr, None)
#                 if child_idx is None:
#                     # might happen if cpr merges with something else or rounding issues
#                     # for simplicity, skip or set to 0
#                     Vs.append(0)
#                 else:
#                     Vs.append(layer[child_idx]["V"])
#             if len(Vs) == 2:
#                 val = disc*(p*Vs[0]+(1-p)*Vs[1])
#             elif len(Vs) == 1:
#                 # degenerate
#                 val = disc*Vs[0]
#             else:
#                 val = rebate
#             nd_prev["V"] = val

#     # finally, the option value at time 0 is nodes[0][0]["V"]
#     return nodes[0][0]["V"]


# # -------------- STREAMLIT DEMO --------------
# import streamlit as st

# def main():
#     st.title("Partially Refined Binomial Tree (Down-and-Out Call)")

#     st.markdown("""
#     This example uses a *partially refined* binomial tree approach:
#     - If a node's price is within a certain 'band' above the barrier, 
#       we refine the tree with extra subdivisions.
#     - Elsewhere, we use a normal CRR approach.
    
#     **Note**: This is purely illustrative and may not be perfectly 
#     arbitrage-free or recombining at boundaries. 
#     It's a compromise between a standard uniform tree and 
#     a fully adaptive (nonuniform) mesh.
#     """)

#     with st.sidebar:
#         st.header("Parameters")
#         S0 = st.number_input("Initial Stock Price (S0)", value=100.0)
#         K  = st.number_input("Strike Price (K)", value=100.0)
#         r  = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.01, format="%.3f")
#         sigma = st.number_input("Volatility (sigma)", value=0.2, step=0.01, format="%.3f")
#         T  = st.number_input("Time to Maturity (T)", value=1.0, step=0.25)
#         N  = st.number_input("Number of Time Steps (N)", value=5, step=1, min_value=1)
        
#         barrier = st.number_input("Down-and-Out Barrier", value=90.0)
#         rebate  = st.number_input("Rebate", value=0.0)
#         refine_band = st.number_input("Refine Band (# of 'price steps' above barrier)", 
#                                       value=2, step=1)

#     if st.button("Calculate Price"):
#         price = partial_refined_binomial_down_and_out_call(
#             S0, K, r, sigma, T,
#             N, barrier, refine_band, rebate
#         )
#         st.write(f"**Down-and-Out Call Price (Partial Refinement):** {price:.4f}")

# if __name__ == "__main__":
#     st.set_page_config(page_title="Partial Refined Binomial Barrier", layout="centered")
#     main()

import streamlit as st
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

# -------------------------------
# 1. Analytical Barrier Option Pricing (Closed-Form)
# -------------------------------
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
    c   = calc_c(S0, K, r, q, sigma, T)
    p   = calc_p(S0, K, r, q, sigma, T)
    lam = calc_lambda(r, q, sigma)
    y   = calc_y(barrier, S0, K, T, sigma, r, q)

    # (The code below is exactly as you provided; handles many cases.)
    # ----------------------------------------------------------------
    # Down-and-in Call
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # Down-and-out Call
    # ----------------------------------------------------------------
    elif option_type == 'down-and-out call' and barrier <= K:
        cdi = (S0 * np.exp(-q*T) * (barrier/S0)**(2*lam) * norm.cdf(y)
               - K * np.exp(-r*T) * (barrier/S0)**(2*lam - 2)
                 * norm.cdf(y - sigma*np.sqrt(T)))
        cdo = c - cdi
        return max(cdo, 0)
    elif option_type == 'down-and-out call' and barrier >= K:
        term1 = S0 * np.exp(-q*T)*norm.cdf(x1)
        term2 = K   * np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
        term3 = S0 * np.exp(-q*T)*((barrier/S0)**(2*lam))*norm.cdf(y1)
        term4 = K   * np.exp(-r*T)*((barrier/S0)**(2*lam - 2))*norm.cdf(y1 - sigma*np.sqrt(T))
        cdo   = term1 - term2 - term3 + term4
        return max(cdo, 0)

    # ----------------------------------------------------------------
    # Up-and-in Call
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # Up-and-out Call
    # ----------------------------------------------------------------
    elif option_type == 'up-and-out call' and barrier <= K:
        # Usually worthless if barrier <= K and S0 is above barrier,
        # but let's keep your logic consistent:
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

    # ----------------------------------------------------------------
    # Up-and-in Put
    # ----------------------------------------------------------------
    elif option_type == 'up-and-in put' and barrier >= K and barrier <= S0:
        pui = black_scholes(S0, K, T, r, sigma, "put")
        return pui
    elif option_type == 'up-and-in put' and barrier >= K:
        pui = (-S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(-y)
               + K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)
                 * norm.cdf(-y + sigma*np.sqrt(T)))
        return pui
    elif option_type == 'up-and-in put' and barrier <= K:
        # up-and-in put = p - up-and-out put in some references,
        # but let's keep your code:
        return p

    # ----------------------------------------------------------------
    # Up-and-out Put
    # ----------------------------------------------------------------
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
            - K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)*norm.cdf(-y1 + sigma*np.sqrt(T))
        )
        return max(puo, 0)

    # ----------------------------------------------------------------
    # Down-and-in Put
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # Down-and-out Put
    # ----------------------------------------------------------------
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


# -------------------------------
# 2. Binomial Tree Pricing (with continuous dividend yield q)
# -------------------------------

def barrier_binomial_option_price(S0, K, r, q_div, T, sigma, steps,
                                  barrier_option_type, H,
                                  option_side='call', rebate=0.0):
    """
    Binomial pricing of a barrier option, tracking whether the barrier is hit.
    Incorporates continuous dividend yield q_div similarly.
    """
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    m = math.exp((r - q_div) * dt)
    p_up = (m - d) / (u - d)

    # Determine barrier direction
    if barrier_option_type.lower().startswith("up"):
        barrier_direction = "up"
    else:
        barrier_direction = "down"
    
    def intrinsic(S):
        if option_side.lower() == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    # Barrier type
    is_knock_out = barrier_option_type.lower().endswith("out")
    is_knock_in  = barrier_option_type.lower().endswith("in")

    memo = {}
    # f(i, j, bh): value at node i, j up-moves so far, bh=barrier hit?
    def f(i, j, bh):
        key = (i, j, bh)
        if key in memo:
            return memo[key]
        
        # Current price
        S = S0 * (u**j) * (d**(i-j))
        t = i * dt
        
        # If at final step
        if i == steps:
            if is_knock_out:
                # If barrier was hit, payoff=rebate; else=intrinsic
                val = rebate if bh else intrinsic(S)
            else:
                # Knock-in: payoff only if barrier was hit
                val = intrinsic(S) if bh else 0.0
            memo[key] = val
            return val
        
        # If knocked out already, payoff=PV of rebate
        if is_knock_out and bh:
            val = rebate * math.exp(-r * (T - t))
            memo[key] = val
            return val

        # Next step up/down
        S_up = S0*(u**(j+1))*(d**((i+1)-(j+1)))
        S_down = S0*(u**j)*(d**((i+1)-j))

        def barrier_hit(S_new, current_bh):
            if current_bh:
                return True
            if barrier_direction == "up":
                return S_new >= H
            else:
                return S_new <= H

        new_bh_up   = barrier_hit(S_up, bh)
        new_bh_down = barrier_hit(S_down, bh)

        # If knock-out and barrier is hit on the next move:
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


def european_binomial_option_price(S0, K, r, q_div, T, sigma, steps, option_type='call'):
    """
    Prices a European option (call or put) using the CRR binomial model
    with continuous dividend yield q_div.
    """
    dt = T / steps
    # Up and down factors
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    # Growth per step = exp((r - q_div)*dt)
    # But discount each step by exp(-r*dt).
    disc = math.exp(-r * dt)
    # Risk-neutral probability
    m = math.exp((r - q_div) * dt)
    p_up = (m - d) / (u - d)

    # Terminal asset prices
    asset_prices = [S0 * (u**j) * (d**(steps - j)) for j in range(steps+1)]
    # Terminal payoffs
    if option_type.lower() == 'call':
        option_values = [max(price - K, 0) for price in asset_prices]
    else:
        option_values = [max(K - price, 0) for price in asset_prices]

    # Backward induction
    for _ in range(steps):
        for i in range(len(option_values)-1):
            option_values[i] = disc * (p_up * option_values[i+1] + (1 - p_up) * option_values[i])
        option_values.pop()

    return option_values[0]


def barrier_binomial_option_price_adaptive(S0, K, r, q_div, T, sigma, steps,
                                            barrier_option_type, H,
                                            option_side='call', rebate=0.0,
                                            barrier_proximity_threshold=0.05):
    """
    Binomial pricing of a barrier option with increased steps near the barrier.
    """
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    m = math.exp((r - q_div) * dt)
    p_up = (m - d) / (u - d)

    # Determine barrier direction
    if barrier_option_type.lower().startswith("up"):
        barrier_direction = "up"
    else:
        barrier_direction = "down"

    def intrinsic(S):
        if option_side.lower() == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    # Barrier type
    is_knock_out = barrier_option_type.lower().endswith("out")
    is_knock_in  = barrier_option_type.lower().endswith("in")

    memo = {}
    # f(i, j, bh): value at node i, j up-moves so far, bh=barrier hit?
    def f(i, j, bh):
        key = (i, j, bh)
        if key in memo:
            return memo[key]

        # Current price
        S = S0 * (u**j) * (d**(i-j))
        t = i * dt

        # If at final step
        if i == steps:
            if is_knock_out:
                # If barrier was hit, payoff=rebate; else=intrinsic
                val = rebate if bh else intrinsic(S)
            else:
                # Knock-in: payoff only if barrier was hit
                val = intrinsic(S) if bh else 0.0
            memo[key] = val
            return val

        # If knocked out already, payoff=PV of rebate
        if is_knock_out and bh:
            val = rebate * math.exp(-r * (T - t))
            memo[key] = val
            return val

        # Next step up/down (standard step)
        S_up = S0*(u**(j+1))*(d**((i+1)-(j+1)))
        S_down = S0*(u**j)*(d**((i+1)-j))

        def barrier_hit(S_new, current_bh):
            if current_bh:
                return True
            if barrier_direction == "up":
                return S_new >= H
            else:
                return S_new <= H

        new_bh_up   = barrier_hit(S_up, bh)
        new_bh_down = barrier_hit(S_down, bh)

        # Check if the price is close to the barrier
        lower_bound = H * (1 - barrier_proximity_threshold)
        upper_bound = H * (1 + barrier_proximity_threshold)
        near_barrier = lower_bound <= S <= upper_bound

        if near_barrier and i < steps -1: # Perform one extra sub-step
            dt_sub = dt / 2
            u_sub = math.exp(sigma * math.sqrt(dt_sub))
            d_sub = 1.0 / u_sub
            disc_sub = math.exp(-r * dt_sub)
            m_sub = math.exp((r - q_div) * dt_sub)
            p_up_sub = (m_sub - d_sub) / (u_sub - d_sub)

            # From S_up, take another step up and down
            S_up_up = S_up * u_sub
            S_up_down = S_up * d_sub
            new_bh_up_up = barrier_hit(S_up_up, new_bh_up)
            new_bh_up_down = barrier_hit(S_up_down, new_bh_up)
            val_up = disc_sub * (f(i + 2, j + 2, new_bh_up_up) * p_up_sub +
                                  f(i + 2, j + 1, new_bh_up_down) * (1 - p_up_sub))

            # From S_down, take another step up and down
            S_down_up = S_down * u_sub
            S_down_down = S_down * d_sub
            new_bh_down_up = barrier_hit(S_down_up, new_bh_down)
            new_bh_down_down = barrier_hit(S_down_down, new_bh_down)
            val_down = disc_sub * (f(i + 2, j + 1, new_bh_down_up) * p_up_sub +
                                    f(i + 2, j, new_bh_down_down) * (1 - p_up_sub))

            val = disc * (val_up * p_up + val_down * (1 - p_up))

        else: # Standard step
            val_up = f(i+1, j+1, new_bh_up)
            val_down = f(i+1, j, new_bh_down)
            val = disc * (p_up * val_up + (1 - p_up) * val_down)

        memo[key] = val
        return val

    return f(0, 0, False)


# -------------------------------
# 3. Enhanced Binomial Tree Visualization
# -------------------------------
def plot_binomial_tree_enhanced(S0, sigma, T, steps, barrier_level=None):
    """
    Plots the binomial tree with color-coded nodes by time step,
    and an optional horizontal barrier line.
    """
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u

    edges_x = []
    edges_y = []
    nodes = []

    for i in range(steps+1):
        for j in range(i+1):
            S = S0 * (u**j)*(d**(i-j))
            nodes.append((i, S))
            if i < steps:
                # up edge
                S_up = S0*(u**(j+1))*(d**((i+1)-(j+1)))
                edges_x += [i, i+1, None]
                edges_y += [S, S_up, None]
                # down edge
                S_down = S0*(u**j)*(d**((i+1)-j))
                edges_x += [i, i+1, None]
                edges_y += [S, S_down, None]

    fig = go.Figure()
    # Plot edges
    fig.add_trace(go.Scatter(x=edges_x, y=edges_y, mode='lines',
                             line=dict(color='gray'), showlegend=False))
    # Plot nodes (color by time step i)
    xs, ys = zip(*nodes)
    node_colors = [n[0] for n in nodes]  # time step = i
    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode='markers',
        marker=dict(size=8, color=node_colors, colorscale='Viridis', showscale=True),
        name='Nodes'
    ))

    if barrier_level is not None:
        fig.add_trace(go.Scatter(
            x=[0, steps],
            y=[barrier_level, barrier_level],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Barrier'
        ))

    fig.update_layout(
        title="Enhanced Binomial Tree of Underlying Asset Prices",
        xaxis_title="Time Step",
        yaxis_title="Asset Price",
        template="simple_white"
    )
    return fig

# Helper to unify barrier type + side for the analytical function
def combine_barrier_and_side(barrier_option_type, side):
    # e.g. "Up-and-Out" + "call" -> "up-and-out call"
    return f"{barrier_option_type.lower().replace(' ', '-')} {side.lower()}"


# -------------------------------
# 4. Streamlit App
# -------------------------------
st.set_page_config(page_title="Option Pricing with Binomial Trees", layout= "wide")
st.title("Option Pricing with Binomial Trees")


# Custom CSS for info boxes
st.markdown(
    """
    <style>
    .info-box {
        background-color: #f9f9f9;
        border-left: 5px solid #007ACC;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        text-align: center;
    }
    .info-box h4 {
        margin: 0;
        color: #007ACC;
    }
    .info-box p {
        margin: 0.5rem 0 0;
        font-size: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True
)
def create_info_box(title, value):
    return f"<div class='info-box'><h4>{title}</h4><p>{value}</p></div>"

st.markdown("""
This app prices both **vanilla European** options and **barrier** options
using a binomial tree. It also compares with an **analytical formula**
(for certain barrier types) and now includes a **conceptual implementation of increased steps near the barrier**.
""")

# Sidebar inputs
with st.sidebar:
    st.header("Model Parameters")
    S0 = st.number_input("Initial Stock Price (S0)", value=100.0, min_value=0.0, step=1.0)
    K   = st.number_input("Strike Price (K)",          value=100.0, min_value=0.0, step=1.0)
    r   = st.number_input("Risk-Free Rate (r)",       value=0.05,  min_value=0.0, step=0.01, format="%.4f")
    q_div = st.number_input("Dividend Yield (q)",    value=0.00,  min_value=0.0, step=0.01, format="%.4f")
    T   = st.number_input("Time to Maturity (T, yrs)", value=1.0, min_value=0.0, step=0.25)
    sigma = st.number_input("Volatility (sigma)",      value=0.2,   min_value=0.0, step=0.01, format="%.4f")
    steps = st.number_input("Binomial Steps",          value=5,     min_value=1,   step=1)

    option_style = st.radio("Option Style", ("Vanilla", "Barrier"))
    option_side   = st.selectbox("Option Side", ["call", "put"])

    # Barrier-specific inputs
    if option_style == "Barrier":
        barrier_option_type = st.selectbox("Barrier Option Type",
                                            ["Up-and-Out", "Down-and-Out", "Up-and-In", "Down-and-In"])
        H = st.number_input("Barrier Level (H)", value=110.0, min_value=0.0, step=1.0)
        rebate = st.number_input("Rebate (if knocked out)", value=0.0, min_value=0.0, step=0.1)
        barrier_proximity = st.number_input("Barrier Proximity Threshold (%)", value=5.0, min_value=0.0, max_value=50.0, step=1.0) / 100.0

# Main area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Vanilla Option Pricing")
    if option_style == "Vanilla":
        if st.button("Calculate Vanilla Option Price"):
            price = european_binomial_option_price(S0, K, r, q_div, T, sigma, steps, option_side)
            st.markdown(create_info_box(f"Binomial {option_side.capitalize()} Option Price", f"{price:.4f}"), unsafe_allow_html=True)

            st.subheader("Binomial Tree Visualization")
            fig = plot_binomial_tree_enhanced(S0, sigma, T, steps)
            st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Barrier Option Pricing")
    if option_style == "Barrier":
        if st.button("Calculate Barrier Option Price (Adaptive Steps Near Barrier)"):
            price_binomial_adaptive = barrier_binomial_option_price_adaptive(S0, K, r, q_div, T, sigma, steps,
                                                                            barrier_option_type, H, option_side, rebate,
                                                                            barrier_proximity_threshold=st.session_state.get('barrier_proximity', 0.05)) # Use session state

            st.markdown(create_info_box(f"Binomial (Adaptive) {barrier_option_type} {option_side.capitalize()} Option Price", f"{price_binomial_adaptive:.4f}"), unsafe_allow_html=True)

        if st.button("Calculate Barrier Option Price (Standard Binomial)"):
            price_binomial = barrier_binomial_option_price(S0, K, r, q_div, T, sigma, steps,
                                                            barrier_option_type, H, option_side, rebate)
            st.markdown(create_info_box(f"Binomial (Standard) {barrier_option_type} {option_side.capitalize()} Option Price", f"{price_binomial:.4f}"), unsafe_allow_html=True)

            # Analytical price for comparison
            analytical_type = combine_barrier_and_side(barrier_option_type, option_side)
            price_analytical = barrier_option_price(S0, K, T, r, q_div, sigma, H, analytical_type)

            if price_analytical is not None:
                st.markdown(create_info_box(f"Analytical {barrier_option_type} {option_side.capitalize()} Option Price", f"{price_analytical:.4f}"), unsafe_allow_html=True)
                error = abs(price_binomial - price_analytical)
                st.markdown(create_info_box("Absolute Error (Standard Binomial)", f"{error:.4f}"), unsafe_allow_html=True)
            else:
                st.warning("Analytical formula not available for this specific barrier option type.")

            st.subheader("Binomial Tree Visualization with Barrier")
            fig = plot_binomial_tree_enhanced(S0, sigma, T, steps, barrier_level=H)
            st.plotly_chart(fig, use_container_width=True)

# Store barrier proximity in session state
if 'barrier_proximity' not in st.session_state:
    st.session_state['barrier_proximity'] = 0.05
with st.sidebar:
    st.session_state['barrier_proximity'] = st.number_input("Barrier Proximity Threshold (%) for Adaptive Steps", value=5.0, min_value=0.0, max_value=50.0, step=1.0) / 100.0

# -------------------------------
# 5. Pricing Barrier Options with Adaptive Mesh (Explanation Updated)
# -------------------------------
st.subheader("Pricing Barrier Options with Increased Steps Near the Barrier")
st.markdown("""
This section implements a simplified approach to increasing the number of time steps specifically when the simulated asset price is close to the barrier.

**Implementation Details:**

When the asset price at a node in the binomial tree falls within a certain percentage range (defined by the 'Barrier Proximity Threshold' in the sidebar) of the barrier level (H), we perform one additional sub-step in the tree. This sub-step uses a time interval that is half of the regular time step, effectively increasing the resolution of the tree around the barrier.

This is a basic form of local refinement. A more sophisticated adaptive mesh would involve dynamically adjusting the number and size of time steps and asset price intervals based on the characteristics of the option price near the barrier.

**How to Use:**

1.  Navigate to the 'Barrier Option Pricing' section.
2.  Ensure 'Barrier' is selected under 'Option Style'.
3.  Set the parameters for the barrier option (Type, Level, Rebate).
4.  Adjust the 'Barrier Proximity Threshold (%) for Adaptive Steps' in the sidebar. A higher percentage means the increased steps will be applied in a wider range around the barrier.
5.  Click the 'Calculate Barrier Option Price (Adaptive Steps Near Barrier)' button to see the price calculated with this local refinement.
6.  For comparison, you can also click the 'Calculate Barrier Option Price (Standard Binomial)' button to see the price without this refinement.

**Note:**

* This adaptive step implementation adds a local increase in the number of steps. It does not change the overall number of steps defined by the 'Binomial Steps' parameter.
* The visualization of the binomial tree in this version still reflects the base number of steps and does not explicitly show the additional sub-steps introduced near the barrier for simplicity. A true visualization of the adaptive tree would be more complex.
* This is a conceptual implementation to illustrate the idea of increasing resolution near the barrier. More advanced adaptive mesh techniques exist for greater accuracy and efficiency.
""")
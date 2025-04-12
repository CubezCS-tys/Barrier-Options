# import streamlit as st
# import numpy as np

# ###############################################################################
# # Trinomial Tree Pricer
# ###############################################################################
# def build_stock_tree(S0, r, sigma, T, N):
#     """
#     Build and return a 2D array of underlying asset prices for a trinomial tree.
#     Each row i corresponds to time step i, and each column j corresponds
#     to the 'node index' at that step (from -i to +i in steps of 1, but we'll
#     store them in 0..2i for convenience).
    
#     S0    : initial stock price
#     r     : risk-free interest rate
#     sigma : volatility
#     T     : time to maturity
#     N     : number of time steps
#     """
#     dt = T / N
#     # Common “standard” choice for up/down factors in a trinomial tree:
#     u = np.exp(sigma * np.sqrt(3 * dt))
#     d = 1.0 / u
#     m = 1.0  # "middle" move is effectively S * m

#     # Create a 2D array (list of lists) for stock prices
#     # At step i, we have (2*i + 1) nodes in a recombining trinomial.
#     stock_tree = []
#     stock_tree.append([S0])  # at time 0, only one node

#     for i in range(1, N + 1):
#         level_prices = []
#         # The middle index at step i is i (0..2i), so:
#         for j in range(2 * i + 1):
#             # The “shift” from the middle node is j - i
#             k = j - i
#             # Price = S0 * u^(#ups) * d^(#downs)
#             # #ups - #downs = k, so #ups = i + k / 2, #downs = i - k / 2 (in a binomial sense)
#             # but simpler is to say: each step can be up, down, or mid.
#             # We can interpret k>0 as net "ups" and k<0 as net "downs".
#             # For a purely multiplicative approach:
#             price = S0 * (u ** max(k, 0)) * (d ** max(-k, 0))
#             level_prices.append(price)
#         stock_tree.append(level_prices)

#     return stock_tree

# def build_barrier_status_tree(stock_tree, barrier_type, barrier_level, N):
#     """
#     Given the stock tree, determine whether each node is 'active' (has not
#     knocked out) or 'inactive' (knocked out), or if we handle knock-in.

#     This function returns a 2D list (same shape as stock_tree) of booleans
#     indicating whether the option at that node is "alive" (True) or "dead" (False).
    
#     For 'knock-out' barriers, once the barrier is touched, we mark that path as dead.
#     For 'knock-in' barriers, we do the opposite (the option only becomes alive if the
#     barrier is touched).  Here we assume discrete monitoring at each time step.

#     barrier_type can be one of:
#       - 'up-and-out'
#       - 'down-and-out'
#       - 'up-and-in'
#       - 'down-and-in'
#       or None if no barrier is used.

#     barrier_level : the barrier price
#     N             : number of time steps
#     """
#     # If no barrier, everything is alive
#     if not barrier_type:
#         return [[True for _ in row] for row in stock_tree]

#     # We'll do a forward pass, checking barrier at each node:
#     # For knock-out, once triggered -> dead. For knock-in, once triggered -> alive.
#     barrier_tree = []
#     barrier_tree.append([False] * len(stock_tree[0]))  # time=0 row
#     # Initialization at time 0:
#     S0 = stock_tree[0][0]
#     if barrier_type == 'up-and-out':
#         # If S0 >= barrier_level, it's knocked out immediately
#         barrier_tree[0][0] = (S0 < barrier_level)
#     elif barrier_type == 'down-and-out':
#         # If S0 <= barrier_level, it's knocked out immediately
#         barrier_tree[0][0] = (S0 > barrier_level)
#     elif barrier_type == 'up-and-in':
#         # If S0 >= barrier_level, it's already in
#         barrier_tree[0][0] = (S0 >= barrier_level)
#     elif barrier_type == 'down-and-in':
#         # If S0 <= barrier_level, it's already in
#         barrier_tree[0][0] = (S0 <= barrier_level)

#     for i in range(1, N + 1):
#         row_alive = []
#         for j in range(len(stock_tree[i])):
#             # We can come from j, j-1, j+1 in the previous row (in principle),
#             # but for simplicity in a recombining tree, let's check direct adjacency:
#             # In practice, you'd carefully map indices. We'll do a broad approach:
#             # parent indices in previous row could be j, j-1, j+1, if valid.
#             parents = []
#             for pj in [j-1, j, j+1]:
#                 if 0 <= pj < len(stock_tree[i-1]):
#                     parents.append(pj)
#             # The node is alive if at least one parent path is alive
#             # AND we haven't triggered a knock-out, or we have triggered a knock-in, etc.
#             current_price = stock_tree[i][j]

#             # Check if this node triggers barrier:
#             if barrier_type in ['up-and-out', 'down-and-out']:
#                 # We want to remain alive only if none of the parents were knocked out
#                 # and we don't cross the barrier now
#                 was_alive = any(barrier_tree[i-1][pj] for pj in parents)
#                 if barrier_type == 'up-and-out':
#                     # Knock out if current_price >= barrier
#                     still_alive = was_alive and (current_price < barrier_level)
#                 else:
#                     # down-and-out
#                     still_alive = was_alive and (current_price > barrier_level)

#             else:
#                 # 'up-and-in' or 'down-and-in'
#                 # We are alive if we have triggered the barrier at any point
#                 was_alive = any(barrier_tree[i-1][pj] for pj in parents)
#                 if barrier_type == 'up-and-in':
#                     # We "become alive" if we cross barrier
#                     # If not crossed barrier yet, remain not alive
#                     # If at this node price >= barrier, then we are definitely in
#                     # Or if the parents were already in.
#                     triggered = (current_price >= barrier_level) or was_alive
#                     still_alive = triggered
#                 else:
#                     # down-and-in
#                     triggered = (current_price <= barrier_level) or was_alive
#                     still_alive = triggered

#             row_alive.append(still_alive)
#         barrier_tree.append(row_alive)

#     return barrier_tree

# def trinomial_option_price(S0, K, r, sigma, T, N,
#                            option_type='call',
#                            option_style='European',
#                            barrier_type=None,
#                            barrier_level=None):
#     """
#     Price a (European/American) call or put option via a trinomial tree.
#     Can also handle simple barrier logic (knock-in or knock-out) if barrier_type
#     is given (and barrier_level is set).
    
#     S0           : initial underlying price
#     K            : strike
#     r            : risk-free rate
#     sigma        : volatility
#     T            : time to maturity
#     N            : number of time steps
#     option_type  : 'call' or 'put'
#     option_style : 'European' or 'American'
#     barrier_type : None, 'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'
#     barrier_level: numeric barrier
#     """
#     dt = T / N
#     # Risk-neutral discount factor
#     discount = np.exp(-r * dt)
    
#     # Up, down, middle factors
#     u = np.exp(sigma * np.sqrt(3 * dt))
#     d = 1.0 / u
#     m = 1.0
    
#     # One common set of risk-neutral probabilities that match the first two moments:
#     # (There are various equivalent formulas in literature. This is a standard one.)
#     # Here is one approach (for simplicity):
#     pu = 1/6 + ( (r - 0.5*sigma**2)*np.sqrt(dt/(12*(sigma**2))))
#     pm = 2/3
#     pd = -( (r - 0.5*sigma**2)*np.sqrt(dt/(12*(sigma**2)))) + 1/6
#     # You might want to ensure pu, pm, pd are valid (>=0) under your chosen parameters.
    
#     # Build the stock price tree
#     stock_tree = build_stock_tree(S0, r, sigma, T, N)
    
#     # Build the barrier status tree (True=alive, False=dead),
#     # for knock-out or knock-in logic
#     barrier_tree = build_barrier_status_tree(stock_tree, barrier_type, barrier_level, N)
    
#     # Initialize option value at maturity
#     # For a barrier option, if the node is not "alive" (for knock-out),
#     # the payoff is 0.  For knock-in, if the node is not "alive" at maturity,
#     # the payoff is 0.
#     payoff_tree = []
#     for i in range(N + 1):
#         payoff_tree.append([0.0]*(2*i + 1))
    
#     # Maturity payoffs
#     for j in range(len(stock_tree[N])):
#         S_T = stock_tree[N][j]
#         alive = barrier_tree[N][j]
#         if alive:
#             if option_type == 'call':
#                 payoff_tree[N][j] = max(S_T - K, 0.0)
#             else:  # put
#                 payoff_tree[N][j] = max(K - S_T, 0.0)
#         else:
#             payoff_tree[N][j] = 0.0  # not alive => no payoff
    
#     # Backward induction
#     for i in range(N-1, -1, -1):
#         for j in range(len(stock_tree[i])):
#             # Only compute if barrier_tree[i][j] is alive
#             if not barrier_tree[i][j]:
#                 payoff_tree[i][j] = 0.0
#                 continue
            
#             # Expected value from next step
#             # child indices in next row: j, j+1, j+2  (because each node i, j
#             # transitions to i+1, j, j+1, j+2 in a recombining tri-tree layout).
#             # But we have to be careful with indexing. The middle child is j+1, up child is j+2, down child is j.
#             # We'll do it carefully:
#             V_u = payoff_tree[i+1][j+2] if (j+2 < len(payoff_tree[i+1])) else 0.0
#             V_m = payoff_tree[i+1][j+1] if (j+1 < len(payoff_tree[i+1])) else 0.0
#             V_d = payoff_tree[i+1][j]   if (j < len(payoff_tree[i+1]))     else 0.0
            
#             # Also must ensure that the barrier_tree is alive for the child nodes
#             # If the child node is dead, that path payoff is 0
#             A_u = barrier_tree[i+1][j+2] if (j+2 < len(barrier_tree[i+1])) else False
#             A_m = barrier_tree[i+1][j+1] if (j+1 < len(barrier_tree[i+1])) else False
#             A_d = barrier_tree[i+1][j]   if (j < len(barrier_tree[i+1]))   else False
            
#             V_expected = pu*(V_u if A_u else 0.0) \
#                          + pm*(V_m if A_m else 0.0) \
#                          + pd*(V_d if A_d else 0.0)
            
#             # Discount back
#             continuation_value = discount * V_expected
            
#             if option_style == 'American':
#                 # Check early exercise
#                 S_ij = stock_tree[i][j]
#                 if option_type == 'call':
#                     exercise_value = max(S_ij - K, 0.0)
#                 else:
#                     exercise_value = max(K - S_ij, 0.0)
#                 payoff_tree[i][j] = max(continuation_value, exercise_value)
#             else:
#                 payoff_tree[i][j] = continuation_value
    
#     # The option price is at the root
#     return payoff_tree[0][0]

# ###############################################################################
# # Streamlit UI
# ###############################################################################
# def main():
#     st.title("Trinomial Tree Option Pricer")

#     st.sidebar.header("Model Parameters")
#     S0 = st.sidebar.number_input("Initial Underlying Price (S0)", value=100.0)
#     K  = st.sidebar.number_input("Strike (K)", value=100.0)
#     r  = st.sidebar.number_input("Risk-free Rate (r)", value=0.05, format="%.5f")
#     sigma = st.sidebar.number_input("Volatility (sigma)", value=0.2, format="%.5f")
#     T  = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0)
#     N  = st.sidebar.number_input("Number of Steps (N)", min_value=1, value=50)

#     st.sidebar.header("Option Type & Style")
#     option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
#     option_style = st.sidebar.selectbox("Option Style", ["European", "American"])

#     st.sidebar.header("Barrier (Optional)")
#     barrier_features = [None, "up-and-out", "down-and-out", "up-and-in", "down-and-in"]
#     barrier_type = st.sidebar.selectbox("Barrier Type", barrier_features, index=0)
#     barrier_level = None
#     if barrier_type is not None and barrier_type != "None":
#         barrier_level = st.sidebar.number_input("Barrier Level", value=120.0)

#     if st.button("Compute Option Price"):
#         price = trinomial_option_price(
#             S0=S0,
#             K=K,
#             r=r,
#             sigma=sigma,
#             T=T,
#             N=int(N),
#             option_type=option_type,
#             option_style=option_style,
#             barrier_type=(barrier_type if barrier_type != "None" else None),
#             barrier_level=barrier_level
#         )
#         st.write(f"### Option Price = {price:0.4f}")

# if __name__ == "__main__":
#     main()


import streamlit as st
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

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
        term2 = K  * np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
        term3 = S0 * np.exp(-q*T)*((barrier/S0)**(2*lam))*norm.cdf(y1)
        term4 = K  * np.exp(-r*T)*((barrier/S0)**(2*lam - 2))*norm.cdf(y1 - sigma*np.sqrt(T))
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

###############################################################################
# 1. Build Trinomial Stock Price Tree
###############################################################################
def build_stock_tree(S0, r, sigma, T, N):
    """
    Build and return a 2D list of underlying asset prices for a recombining
    trinomial tree. Each row i (0 <= i <= N) contains (2*i + 1) nodes.
    
    S0    : initial stock price
    r     : risk-free rate (not used here; prices are based on volatility only)
    sigma : volatility
    T     : time to maturity
    N     : number of time steps
    """
    dt = T / N
    # In a trinomial tree using the "standard" approach, we set:
    u = np.exp(sigma * np.sqrt(3 * dt))
    d = 1.0 / u
    # Middle move factor is 1 (i.e. no change)
    
    stock_tree = []
    stock_tree.append([S0])  # time 0: one node

    for i in range(1, N + 1):
        level_prices = []
        # There are (2*i + 1) nodes at step i.
        for j in range(2 * i + 1):
            # Define k = j - i (so that the middle node has k = 0)
            k = j - i
            # A simple multiplicative approach: if k>0, count as net up moves; if k<0, as net down moves.
            price = S0 * (u ** max(k, 0)) * (d ** max(-k, 0))
            level_prices.append(price)
        stock_tree.append(level_prices)
    return stock_tree

###############################################################################
# 2. Build Barrier Status Tree
###############################################################################
def build_barrier_status_tree(stock_tree, barrier_type, barrier_level, N):
    """
    Given the stock tree, return a 2D list (same shape) of booleans indicating
    whether each node is “alive” (True) or “dead” (False) according to the barrier.
    
    For knock–out options, once the barrier is hit the node is marked as dead.
    For knock–in options, a node is alive only if the barrier has been hit on that path.
    """
    # If no barrier provided, all nodes are alive.
    if (not barrier_type) or (barrier_level is None):
        return [[True for _ in row] for row in stock_tree]
    
    barrier_tree = []
    # Time 0: one node
    barrier_tree.append([False])
    S0 = stock_tree[0][0]
    # Initialize time 0 based on barrier type:
    if barrier_type == 'up-and-out':
        barrier_tree[0][0] = (S0 < barrier_level)
    elif barrier_type == 'down-and-out':
        barrier_tree[0][0] = (S0 > barrier_level)
    elif barrier_type == 'up-and-in':
        barrier_tree[0][0] = (S0 >= barrier_level)
    elif barrier_type == 'down-and-in':
        barrier_tree[0][0] = (S0 <= barrier_level)
    
    N_steps = len(stock_tree) - 1
    for i in range(1, N_steps + 1):
        row_alive = []
        for j in range(len(stock_tree[i])):
            # Consider possible parents: indices j-1, j, j+1 in previous row.
            parents = []
            for pj in [j-1, j, j+1]:
                if 0 <= pj < len(stock_tree[i-1]):
                    parents.append(pj)
            current_price = stock_tree[i][j]
            was_alive = any(barrier_tree[i-1][pj] for pj in parents)
            if barrier_type in ['up-and-out', 'down-and-out']:
                # For knock-out: remain alive only if parents were alive and current price is on the proper side.
                if barrier_type == 'up-and-out':
                    still_alive = was_alive and (current_price < barrier_level)
                else:  # down-and-out
                    still_alive = was_alive and (current_price > barrier_level)
            else:
                # For knock-in: become alive if any parent was already in or if the current price triggers barrier.
                if barrier_type == 'up-and-in':
                    triggered = (current_price >= barrier_level) or was_alive
                else:  # down-and-in
                    triggered = (current_price <= barrier_level) or was_alive
                still_alive = triggered
            row_alive.append(still_alive)
        barrier_tree.append(row_alive)
    return barrier_tree

###############################################################################
# 3. Trinomial Option Pricing (European & Barrier)
###############################################################################
def trinomial_option_price(S0, K, r, sigma, T, N,
                           option_type='call',
                           option_style='European',
                           barrier_type=None,
                           barrier_level=None):
    """
    Prices a European (or American) call or put option using a trinomial tree.
    Can also handle barrier options (knock-in or knock-out).
    
    S0           : initial stock price
    K            : strike price
    r            : risk-free rate
    sigma        : volatility
    T            : time to maturity
    N            : number of time steps
    option_type  : 'call' or 'put'
    option_style : 'European' or 'American'
    barrier_type : e.g., 'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'
    barrier_level: barrier price
    """
    dt = T / N
    discount = np.exp(-r * dt)
    
    # Trinomial factors (standard approach)
    u = np.exp(sigma * np.sqrt(3 * dt))
    d = 1.0 / u
    # Set probabilities that match the first two moments.
    pu = 1/6 + ((r - 0.5*sigma**2) * np.sqrt(dt/(12*(sigma**2))))
    pm = 2/3
    pd = 1 - pu - pm
    # Clamp to nonnegative and renormalize if needed:
    pu = max(pu, 0)
    pd = max(pd, 0)
    total_p = pu + pm + pd
    if total_p > 0:
        pu /= total_p; pm /= total_p; pd /= total_p
    
    # Build stock price tree and barrier status tree
    stock_tree = build_stock_tree(S0, r, sigma, T, N)
    barrier_tree = build_barrier_status_tree(stock_tree, barrier_type, barrier_level, N)
    
    # Initialize payoff tree (same shape as stock_tree)
    payoff_tree = []
    for i in range(N + 1):
        payoff_tree.append([0.0]*(2*i + 1))
    
    # Set terminal payoffs
    for j in range(len(stock_tree[N])):
        ST = stock_tree[N][j]
        alive = barrier_tree[N][j]
        if alive:
            if option_type.lower() == 'call':
                payoff_tree[N][j] = max(ST - K, 0.0)
            else:
                payoff_tree[N][j] = max(K - ST, 0.0)
        else:
            payoff_tree[N][j] = 0.0  # If knocked out or barrier never triggered (for knock-in)
    
    # Backward induction through the tree
    for i in range(N-1, -1, -1):
        for j in range(len(stock_tree[i])):
            if not barrier_tree[i][j]:
                payoff_tree[i][j] = 0.0
                continue
            
            # In a recombining trinomial tree, each node at (i,j) has three children:
            # down: index j, middle: index j+1, up: index j+2 in row i+1.
            V_d = payoff_tree[i+1][j]   if (j < len(payoff_tree[i+1])) else 0.0
            V_m = payoff_tree[i+1][j+1] if (j+1 < len(payoff_tree[i+1])) else 0.0
            V_u = payoff_tree[i+1][j+2] if (j+2 < len(payoff_tree[i+1])) else 0.0
            
            # Also check barrier status for children
            A_d = barrier_tree[i+1][j]   if (j < len(barrier_tree[i+1])) else False
            A_m = barrier_tree[i+1][j+1] if (j+1 < len(barrier_tree[i+1])) else False
            A_u = barrier_tree[i+1][j+2] if (j+2 < len(barrier_tree[i+1])) else False
            
            V_expected = pu * (V_u if A_u else 0.0) \
                         + pm * (V_m if A_m else 0.0) \
                         + pd * (V_d if A_d else 0.0)
            cont_value = discount * V_expected
            
            if option_style == 'American':
                # Early exercise check
                S_ij = stock_tree[i][j]
                if option_type.lower() == 'call':
                    exercise_value = max(S_ij - K, 0.0)
                else:
                    exercise_value = max(K - S_ij, 0.0)
                payoff_tree[i][j] = max(cont_value, exercise_value)
            else:
                payoff_tree[i][j] = cont_value
    return payoff_tree[0][0]

###############################################################################
# 4. Helper: Combine Barrier Type and Option Side
###############################################################################
def combine_barrier_and_side(barrier_option_type, side):
    # e.g., "Up-and-Out" + "call" -> "up-and-out call"
    return f"{barrier_option_type.lower().replace(' ', '-')} {side.lower()}"

###############################################################################
# 5. Visualization: Trinomial Tree (Enhanced)
###############################################################################
def plot_trinomial_tree_enhanced(S0, sigma, T, steps, barrier_level=None):
    """
    Plots the trinomial tree of asset prices.
    Each node is computed as: S0 * u^(max(k,0)) * d^(max(-k,0)),
    where k = j - i for row i (with 2*i+1 nodes).
    """
    dt = T / steps
    u = math.exp(sigma * math.sqrt(3 * dt))
    d = 1.0 / u

    edges_x = []
    edges_y = []
    nodes = []

    for i in range(steps + 1):
        for j in range(2 * i + 1):
            S = S0 * (u ** max(j - i, 0)) * (d ** max(i - j, 0))
            nodes.append((i, S))
            if i < steps:
                # Each node (i,j) leads to three nodes at i+1:
                S_down = S0 * (u ** max(j - i, 0)) * (d ** max(i - j + 1, 0))
                S_mid  = S0 * (u ** max(j - i + 1, 0)) * (d ** max(i - j, 0))
                S_up   = S0 * (u ** max(j - i + 2, 0)) * (d ** max(i - j - 1, 0))
                # Draw edges (down, mid, up)
                edges_x += [i, i+1, None]
                edges_y += [S, S_down, None]
                edges_x += [i, i+1, None]
                edges_y += [S, S_mid, None]
                edges_x += [i, i+1, None]
                edges_y += [S, S_up, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edges_x, y=edges_y, mode='lines',
                             line=dict(color='gray'), showlegend=False))
    xs, ys = zip(*nodes)
    node_colors = [n[0] for n in nodes]  # color by time step
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers',
                             marker=dict(size=8, color=node_colors, colorscale='Viridis', showscale=True),
                             name='Nodes'))
    if barrier_level is not None:
        fig.add_trace(go.Scatter(x=[0, steps], y=[barrier_level, barrier_level],
                                 mode='lines', line=dict(color='red', dash='dash'),
                                 name='Barrier'))
    fig.update_layout(title="Enhanced Trinomial Tree of Asset Prices",
                      xaxis_title="Time Step", yaxis_title="Asset Price",
                      template="simple_white")
    return fig

###############################################################################
# 6. Streamlit App Interface
###############################################################################
st.set_page_config(page_title="Option Pricing with Trinomial Trees", layout="wide")
st.title("Option Pricing with Trinomial Trees")

st.markdown("""
This app prices both **vanilla European** options and **barrier** options 
using a **trinomial tree**. It also compares with an **analytical formula** 
(for certain barrier types) and shows **error analysis**.
""")

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

# Sidebar Inputs
with st.sidebar:
    st.header("Model Parameters")
    S0 = st.number_input("Initial Stock Price (S0)", value=100.0, min_value=0.0, step=1.0)
    K  = st.number_input("Strike Price (K)", value=100.0, min_value=0.0, step=1.0)
    r  = st.number_input("Risk-Free Rate (r)", value=0.05, min_value=0.0, step=0.01, format="%.4f")
    q_div = st.number_input("Dividend Yield (q)", value=0.00, min_value=0.0, step=0.01, format="%.4f")
    T  = st.number_input("Time to Maturity (T, yrs)", value=1.0, min_value=0.0, step=0.25)
    sigma = st.number_input("Volatility (sigma)", value=0.2, min_value=0.0, step=0.01, format="%.4f")
    steps = st.number_input("Number of Steps in Trinomial Tree", value=5, min_value=1, step=1)
    
    option_style = st.radio("Option Style", ("Vanilla", "Barrier"))
    option_side  = st.selectbox("Option Side", ["call", "put"])
    
    # Barrier-specific inputs
    if option_style == "Barrier":
        barrier_option_type = st.selectbox("Barrier Option Type", ["Up-and-Out", "Down-and-Out", "Up-and-In", "Down-and-In"])
        H = st.number_input("Barrier Level (H)", value=110.0, min_value=0.0, step=1.0)
        rebate = st.number_input("Rebate (if knocked out)", value=0.0, min_value=0.0, step=0.1)

# Main Panel
if st.button("Calculate Option Price"):
    colA, colB, colC = st.columns(3)
    if option_style == "Vanilla":
        price_trinomial = trinomial_option_price(S0, K, r, sigma, T, steps, option_type=option_side, option_style=option_style)
        st.markdown(create_info_box(f"Vanilla {option_side.capitalize()} Option (Trinomial)", f"${price_trinomial:.4f}"), unsafe_allow_html=True)
        bs_price = black_scholes(S0, K, T, r, sigma, option_side)
        st.markdown(create_info_box(f"Vanilla {option_side.capitalize()} Option (Analytical BS)", f"${bs_price:.4f}"), unsafe_allow_html=True)
        st.markdown(create_info_box("Difference", f"{price_trinomial - bs_price:.4f}"), unsafe_allow_html=True)
    else:
        price_trinomial = trinomial_option_price(S0, K, r, sigma, T, steps, option_type=option_side, option_style=option_style, barrier_type=barrier_option_type, barrier_level=H)
        st.markdown(create_info_box(f"{barrier_option_type} {option_side.capitalize()} Option (Trinomial)", f"${price_trinomial:.4f}"), unsafe_allow_html=True)
        cf_type = combine_barrier_and_side(barrier_option_type, option_side)
        price_analytic = barrier_option_price(S0, K, T, r, q_div, sigma, H, cf_type)
        if price_analytic is not None:
            st.markdown(create_info_box(f"{barrier_option_type} {option_side.capitalize()} (Analytical)", f"${price_analytic:.4f}"), unsafe_allow_html=True)
            st.markdown(create_info_box("Difference (Trinomial - Analytical)", f"{price_trinomial - price_analytic:.4f}"), unsafe_allow_html=True)
        else:
            st.warning("Analytical formula returned None for these parameters.")
    
    # Tree Visualization
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Trinomial Tree Visualization")
        if option_style == "Vanilla":
            fig = plot_trinomial_tree_enhanced(S0, sigma, T, steps, barrier_level=None)
        else:
            fig = plot_trinomial_tree_enhanced(S0, sigma, T, steps, barrier_level=H)
        st.plotly_chart(fig, use_container_width=True)
    
    # Optional Error Analysis for Barrier Options (using trinomial steps)
    if option_style == "Barrier":
        with col2:
            cf_type = combine_barrier_and_side(barrier_option_type, option_side)
            analytic_price = barrier_option_price(S0, K, T, r, q_div, sigma, H, cf_type)
            if analytic_price is None:
                st.warning("No valid analytical price for these parameters.")
            else:
                max_steps_for_error = st.slider("Max steps for error analysis", 1, 400, 10)
                data_points = []
                for n in range(1, max_steps_for_error + 1):
                    trin_price = trinomial_option_price(S0, K, r, sigma, T, n, option_type=option_side, option_style=option_style, barrier_type=barrier_option_type, barrier_level=H)
                    err = trin_price - analytic_price
                    data_points.append({"Steps": n, "Trinomial Price": trin_price, "Error": err})
                df_error = pd.DataFrame(data_points)
                fig_conv = go.Figure()
                fig_conv.add_trace(go.Scatter(
                    x=df_error["Steps"], y=df_error["Trinomial Price"],
                    mode='lines+markers', name='Trinomial Price'
                ))
                fig_conv.add_shape(type='line',
                                   x0=1, x1=max_steps_for_error,
                                   y0=analytic_price, y1=analytic_price,
                                   line=dict(color='red', dash='dash'))
                fig_conv.add_annotation(x=max_steps_for_error, y=analytic_price,
                                        xanchor='left', text=f"Analytical = {analytic_price:.4f}",
                                        showarrow=False, font=dict(color='red'))
                fig_conv.update_layout(title="Convergence of Trinomial Price to Analytical Price",
                                       xaxis_title="Steps", yaxis_title="Price")
                st.plotly_chart(fig_conv, use_container_width=True)

if option_style == "Barrier":
    st.subheader("Error vs. Stock Price (for Barrier Options)")
    S0_min = st.number_input("Minimum S0", value=50.0, step=1.0)
    S0_max = st.number_input("Maximum S0", value=150.0, step=1.0)
    S0_step = st.number_input("Increment for S0", value=5.0, step=1.0)
    steps_for_error = st.number_input("Number of Trinomial Steps for Error Analysis", value=50, min_value=1)
    if st.button("Plot Error vs. Stock Price"):
        S0_values = np.arange(S0_min, S0_max + S0_step, S0_step)
        errors = []
        for s in S0_values:
            trin_price = trinomial_option_price(s, K, r, sigma, T, steps_for_error, option_type=option_side, option_style=option_style, barrier_type=barrier_option_type, barrier_level=H)
            cf_type = combine_barrier_and_side(barrier_option_type, option_side)
            an_price = barrier_option_price(s, K, T, r, q_div, sigma, H, cf_type)
            err = trin_price - an_price if an_price is not None else np.nan
            errors.append(err)
        fig_err_s0 = go.Figure()
        fig_err_s0.add_trace(go.Scatter(
            x=S0_values, y=errors,
            mode='lines+markers', name='Error (Trinomial - Analytical)'
        ))
        fig_err_s0.update_layout(title="Error vs. Stock Price (S0)",
                                 xaxis_title="Stock Price (S0)", yaxis_title="Error",
                                 template="simple_white")
        st.plotly_chart(fig_err_s0, use_container_width=True)

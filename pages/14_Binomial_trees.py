# # import streamlit as st
# # import math

# # def european_binomial_option_price(S0, K, r, T, sigma, steps, option_type='call'):
# #     """
# #     Prices a European option (call or put) using the Cox-Ross-Rubinstein binomial model.
    
# #     Parameters:
# #     -----------
# #     S0 : float
# #         Current underlying asset price
# #     K : float
# #         Strike price of the option
# #     r : float
# #         Risk-free interest rate (annualized, as a decimal)
# #     T : float
# #         Time to maturity in years
# #     sigma : float
# #         Volatility of the underlying asset (annualized, as a decimal)
# #     steps : int
# #         Number of time steps in the binomial model
# #     option_type : str
# #         'call' or 'put'
    
# #     Returns:
# #     --------
# #     float
# #         The price of the option
# #     """
# #     # Time step
# #     dt = T / steps
    
# #     # Cox-Ross-Rubinstein up & down factors
# #     u = math.exp(sigma * math.sqrt(dt))  # up factor
# #     d = 1.0 / u                          # down factor
    
# #     # Risk-neutral probability
# #     R = math.exp(r * dt)         # growth per step at risk-free rate
# #     q = (R - d) / (u - d)        # risk-neutral up probability
    
# #     # 1) Compute asset prices at maturity for all possible paths
# #     #    (the last layer of the binomial tree).
# #     asset_prices = [(S0 * (u ** j) * (d ** (steps - j))) for j in range(steps + 1)]
    
# #     # 2) Compute option payoffs at maturity
# #     if option_type.lower() == 'call':
# #         option_values = [max(0.0, price - K) for price in asset_prices]
# #     else:  # put
# #         option_values = [max(0.0, K - price) for price in asset_prices]
        
# #     # 3) Step back through the binomial tree
# #     #    Discount option values at each node to get today's price
# #     for _ in range(steps):
# #         for i in range(len(option_values) - 1):
# #             # Expected option value under risk-neutral measure, discounted back
# #             option_values[i] = (1/R) * (q * option_values[i+1] + (1 - q) * option_values[i])
# #         # At each step, the list of option values gets shorter by 1
# #         option_values.pop()
    
# #     # The first element now contains the option price at t=0
# #     return option_values[0]

# # # Streamlit App
# # def main():
# #     st.title("European Option Pricing with the Binomial Tree Method")

# #     st.markdown("""
# #     This Streamlit app calculates the **European Call and Put Option** prices 
# #     using the **Cox-Ross-Rubinstein (CRR) Binomial Tree** model.
# #     """)

# #     with st.sidebar:
# #         st.header("Model Parameters")
# #         S0 = st.number_input("Initial Stock Price (S0)", value=100.0, min_value=0.0, step=1.0)
# #         K = st.number_input("Strike Price (K)", value=100.0, min_value=0.0, step=1.0)
# #         r = st.number_input("Risk-Free Rate (r)", value=0.05, min_value=0.0, step=0.01, format="%.4f")
# #         T = st.number_input("Time to Maturity (T in years)", value=1.0, min_value=0.0, step=0.25)
# #         sigma = st.number_input("Volatility (sigma)", value=0.2, min_value=0.0, step=0.01, format="%.4f")
# #         steps = st.number_input("Number of Steps in Binomial Tree", value=3, min_value=1, step=1)
        
# #         option_type = st.selectbox("Option Type", ["call", "put"])

# #     if st.button("Calculate Option Price"):
# #         price = european_binomial_option_price(S0, K, r, T, sigma, steps, option_type)
# #         st.write(f"**The {option_type.capitalize()} option price is:** {price:.4f}")

# # if __name__ == '__main__':
# #     main()


# import streamlit as st
# import math
# import numpy as np
# import plotly.graph_objects as go

# # -------------------------------
# # 1. Vanilla European Option Pricing
# # -------------------------------
# def european_binomial_option_price(S0, K, r, T, sigma, steps, option_type='call'):
#     """
#     Prices a European option (call or put) using the Cox-Ross-Rubinstein binomial model.
#     """
#     dt = T / steps
#     u = math.exp(sigma * math.sqrt(dt))  # up factor
#     d = 1.0 / u                          # down factor
#     R = math.exp(r * dt)                 # risk-free growth per step
#     q = (R - d) / (u - d)                # risk-neutral probability
    
#     # Terminal asset prices at maturity
#     asset_prices = [S0 * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)]
    
#     # Terminal payoffs
#     if option_type.lower() == 'call':
#         option_values = [max(price - K, 0) for price in asset_prices]
#     else:
#         option_values = [max(K - price, 0) for price in asset_prices]
        
#     # Backward induction through the tree
#     for _ in range(steps):
#         for i in range(len(option_values) - 1):
#             option_values[i] = (1/R) * (q * option_values[i+1] + (1 - q) * option_values[i])
#         option_values.pop()
    
#     return option_values[0]

# # -------------------------------
# # 2. Barrier Option Pricing via Binomial Tree (Recursive)
# # -------------------------------
# def barrier_binomial_option_price(S0, K, r, T, sigma, steps, barrier_option_type, H, option_side='call', rebate=0.0):
#     """
#     Prices a barrier option using a binomial tree that tracks whether the barrier has been hit.
    
#     Parameters:
#       - S0, K, r, T, sigma, steps: as before.
#       - barrier_option_type: one of "Up-and-Out", "Down-and-Out", "Up-and-In", "Down-and-In"
#       - H: barrier level
#       - option_side: "call" or "put"
#       - rebate: rebate paid if the option is knocked out (assumed paid at maturity)
    
#     Returns:
#       - Option price.
#     """
#     dt = T / steps
#     u = math.exp(sigma * math.sqrt(dt))
#     d = 1.0 / u
#     R = math.exp(r * dt)
#     q = (R - d) / (u - d)
    
#     # Determine barrier direction ("up" or "down") from the barrier_option_type.
#     if barrier_option_type.lower().startswith("up"):
#         barrier_direction = "up"
#     elif barrier_option_type.lower().startswith("down"):
#         barrier_direction = "down"
#     else:
#         raise ValueError("Invalid barrier option type.")
        
#     # Define intrinsic payoff (for European options)
#     def intrinsic(S):
#         if option_side.lower() == 'call':
#             return max(S - K, 0)
#         else:
#             return max(K - S, 0)
    
#     # Helper: update barrier-hit flag based on new asset price S_new.
#     def update_bh(S_new, current_bh):
#         if current_bh:
#             return True
#         if barrier_direction == "up":
#             return S_new >= H
#         else:
#             return S_new <= H

#     # For knock-out options, once barrier is hit the option becomes "dead".
#     # We assume the rebate (if any) is paid at maturity; thus, if knocked out at time t,
#     # the value is: rebate * exp(-r*(T-t)).
#     is_knock_out = barrier_option_type.lower().endswith("out")
#     is_knock_in  = barrier_option_type.lower().endswith("in")
    
#     memo = {}
#     # f(i, j, bh): value at node corresponding to step i, having taken j up moves,
#     # with bh indicating whether the barrier has been hit on the path.
#     def f(i, j, bh):
#         key = (i, j, int(bh))
#         if key in memo:
#             return memo[key]
#         # Current time and asset price:
#         t = i * dt
#         S = S0 * (u ** j) * (d ** (i - j))
        
#         # Base case: at maturity
#         if i == steps:
#             if is_knock_out:
#                 # For knock-out: if barrier was hit, payoff is rebate; else, normal intrinsic payoff.
#                 value = rebate if bh else intrinsic(S)
#             else:  # knock-in
#                 # For knock-in: option only exists if barrier has been hit.
#                 value = intrinsic(S) if bh else 0.0
#             memo[key] = value
#             return value
        
#         # For knock-out options, if already knocked out, terminate early:
#         if is_knock_out and bh:
#             value = rebate * math.exp(-r * (T - t))
#             memo[key] = value
#             return value
        
#         # Otherwise, continue to next step.
#         # Compute the value for an up move:
#         S_up = S0 * (u ** (j + 1)) * (d ** ((i + 1) - (j + 1)))  # = S0 * u^(j+1)*d^(i-j)
#         new_bh_up = update_bh(S_up, bh)
#         # Compute the value for a down move:
#         S_down = S0 * (u ** j) * (d ** ((i + 1) - j))
#         new_bh_down = update_bh(S_down, bh)
        
#         # For knock-out options, if a move causes a barrier breach, we can terminate that branch:
#         if is_knock_out and new_bh_up:
#             value_up = rebate * math.exp(-r * (T - (t + dt)))
#         else:
#             value_up = f(i + 1, j + 1, new_bh_up)
            
#         if is_knock_out and new_bh_down:
#             value_down = rebate * math.exp(-r * (T - (t + dt)))
#         else:
#             value_down = f(i + 1, j, new_bh_down)
        
#         value = (1/R) * (q * value_up + (1 - q) * value_down)
#         memo[key] = value
#         return value

#     return f(0, 0, False)

# # -------------------------------
# # 3. Visualization of the Binomial Tree
# # -------------------------------
# def plot_binomial_tree(S0, sigma, T, steps, barrier_level=None):
#     """
#     Plots the asset price binomial tree. Each node is computed as S0 * u^j * d^(i-j),
#     where dt = T/steps, u = exp(sigma*sqrt(dt)) and d = 1/u.
#     If barrier_level is provided, a horizontal line is added.
#     """
#     dt = T / steps
#     u = math.exp(sigma * math.sqrt(dt))
#     d = 1.0 / u
    
#     # Collect nodes and edges
#     nodes = []
#     edges_x = []
#     edges_y = []
    
#     for i in range(steps + 1):
#         for j in range(i + 1):
#             S = S0 * (u ** j) * (d ** (i - j))
#             nodes.append((i, S))
#             # Create edges from current node to next nodes (if not at final step)
#             if i < steps:
#                 # Up move
#                 nodes_up = (i + 1, S0 * (u ** (j + 1)) * (d ** ((i + 1) - (j + 1))))
#                 edges_x.extend([i, i + 1, None])
#                 edges_y.extend([S, nodes_up[1], None])
#                 # Down move
#                 nodes_down = (i + 1, S0 * (u ** j) * (d ** ((i + 1) - j)))
#                 edges_x.extend([i, i + 1, None])
#                 edges_y.extend([S, nodes_down[1], None])
    
#     fig = go.Figure()
#     # Plot the edges (tree lines)
#     fig.add_trace(go.Scatter(x=edges_x, y=edges_y, mode='lines', line=dict(color='gray'), showlegend=False))
#     # Plot the nodes
#     xs, ys = zip(*nodes)
#     fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers', marker=dict(size=8, color='black'), name='Nodes'))
    
#     # Add barrier level line if provided
#     if barrier_level is not None:
#         fig.add_trace(go.Scatter(
#             x=[0, steps],
#             y=[barrier_level, barrier_level],
#             mode='lines',
#             line=dict(color='red', dash='dash'),
#             name='Barrier Level'
#         ))
    
#     fig.update_layout(
#         title="Binomial Tree of Underlying Asset Prices",
#         xaxis_title="Time Step",
#         yaxis_title="Asset Price",
#         template="simple_white"
#     )
#     return fig

# # -------------------------------
# # 4. Streamlit App Interface
# # -------------------------------
# def main():
#     st.title("Option Pricing with Binomial Trees")
    
#     st.markdown("""
#     This app prices European options via the CRR binomial tree method. 
#     It can also price barrier options (up-and-out, down-and-out, up-and-in, down-and-in) 
#     for both calls and puts. A visualization of the underlying binomial tree is shown below.
#     """)
    
#     with st.sidebar:
#         st.header("Model Parameters")
#         S0 = st.number_input("Initial Stock Price (S0)", value=100.0, min_value=0.0, step=1.0)
#         K = st.number_input("Strike Price (K)", value=100.0, min_value=0.0, step=1.0)
#         r = st.number_input("Risk-Free Rate (r)", value=0.05, min_value=0.0, step=0.01, format="%.4f")
#         T = st.number_input("Time to Maturity (T in years)", value=1.0, min_value=0.0, step=0.25)
#         sigma = st.number_input("Volatility (sigma)", value=0.2, min_value=0.0, step=0.01, format="%.4f")
#         steps = st.number_input("Number of Steps in the Binomial Tree", value=5, min_value=1, step=1)
        
#         option_style = st.radio("Select Option Style", ("Vanilla", "Barrier"))
#         option_side = st.selectbox("Option Side", ["call", "put"])
    
#         if option_style == "Vanilla":
#             st.info("Pricing a standard European option using the binomial tree.")
#         else:
#             barrier_option_type = st.selectbox("Barrier Option Type", 
#                                                  ["Up-and-Out", "Down-and-Out", "Up-and-In", "Down-and-In"])
#             H = st.number_input("Barrier Level (H)", value=110.0, min_value=0.0, step=1.0)
#             rebate = st.number_input("Rebate (if knocked out)", value=0.0, min_value=0.0, step=0.1)
    
#     if st.button("Calculate Option Price"):
#         if option_style == "Vanilla":
#             price = european_binomial_option_price(S0, K, r, T, sigma, steps, option_side)
#             st.write(f"**Vanilla {option_side.capitalize()} Option Price:** {price:.4f}")
#         else:
#             price = barrier_binomial_option_price(S0, K, r, T, sigma, steps, barrier_option_type, H, option_side, rebate)
#             st.write(f"**{barrier_option_type} {option_side.capitalize()} Barrier Option Price:** {price:.4f}")
    
#     st.subheader("Binomial Tree Visualization")
#     if option_style == "Vanilla":
#         fig = plot_binomial_tree(S0, sigma, T, steps)
#     else:
#         fig = plot_binomial_tree(S0, sigma, T, steps, barrier_level=H)
#     st.plotly_chart(fig, use_container_width=True)

# if __name__ == '__main__':
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


# -------------------------------
# 2. Binomial Tree Pricing (with continuous dividend yield q)
# -------------------------------
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
(for certain barrier types) and shows **error analysis**.
""")

# Sidebar inputs
with st.sidebar:
    st.header("Model Parameters")
    S0 = st.number_input("Initial Stock Price (S0)", value=100.0, min_value=0.0, step=1.0)
    K  = st.number_input("Strike Price (K)",         value=100.0, min_value=0.0, step=1.0)
    r  = st.number_input("Risk-Free Rate (r)",      value=0.05,  min_value=0.0, step=0.01, format="%.4f")
    q_div = st.number_input("Dividend Yield (q)",   value=0.00,  min_value=0.0, step=0.01, format="%.4f")
    T  = st.number_input("Time to Maturity (T, yrs)", value=1.0, min_value=0.0, step=0.25)
    sigma = st.number_input("Volatility (sigma)",   value=0.2,  min_value=0.0, step=0.01, format="%.4f")
    steps = st.number_input("Binomial Steps",       value=5,    min_value=1,   step=1)

    option_style = st.radio("Option Style", ("Vanilla", "Barrier"))
    option_side  = st.selectbox("Option Side", ["call", "put"])

    # Barrier-specific inputs
    if option_style == "Barrier":
        barrier_option_type = st.selectbox("Barrier Option Type", 
            ["Up-and-Out", "Down-and-Out", "Up-and-In", "Down-and-In"])
        H = st.number_input("Barrier Level (H)", value=110.0, min_value=0.0, step=1.0)
        rebate = st.number_input("Rebate (if knocked out)", value=0.0, min_value=0.0, step=0.1)

# Main panel
if st.button("Calculate Option Price"):
    colA, colB, colC = st.columns(3)
    if option_style == "Vanilla":
        with colA:
            price_binomial = european_binomial_option_price(
                S0, K, r, q_div, T, sigma, steps, option_side
            )
            #st.write(f"**Vanilla {option_side.capitalize()} Option (Binomial) Price:** {price_binomial:.4f}")
            st.markdown(create_info_box(f"Vanilla {option_side.capitalize()} Option (Binomial) Price", f"${price_binomial:.4f}"), unsafe_allow_html=True)
        with colB:
            # Compare with standard Black-Scholes
            bs_price = black_scholes(S0, K, T, r, sigma, option_side)
            #st.write(f"**Vanilla {option_side.capitalize()} Option (Analytical BS) Price:** {bs_price:.4f}")
            st.markdown(create_info_box(f"Vanilla {option_side.capitalize()} Option (Analytical BS) Price", f"${bs_price:.4f}"), unsafe_allow_html=True)
        with colC:
            #st.write(f"**Difference (Binomial - BS):** {price_binomial - bs_price:.4e}")
            st.markdown(create_info_box(f"Difference", f"{price_binomial - bs_price:.4f}"), unsafe_allow_html=True)

    else:
        # Barrier via Binomial
        with colA:
            price_binomial = barrier_binomial_option_price(
                S0, K, r, q_div, T, sigma, steps, barrier_option_type, H,
                option_side, rebate
            )
            #st.write(f"**{barrier_option_type} {option_side.capitalize()} (Binomial) Price:** {price_binomial:.4f}")
            st.markdown(create_info_box(f"{barrier_option_type} {option_side.capitalize()} (Binomial) Price", f"${price_binomial:.4f}", ), unsafe_allow_html=True)
        # Barrier via Closed-Form (if it applies)
        # We'll combine the type: e.g. "Up-and-Out" + "call" -> "up-and-out call"

        cf_type = combine_barrier_and_side(barrier_option_type, option_side)
        price_analytic = barrier_option_price(S0, K, T, r, q_div, sigma, H, cf_type)
        if price_analytic is not None:
            with colB:
                #st.write(f"**{barrier_option_type} {option_side.capitalize()} (Analytical) Price:** {price_analytic:.4f}")
                st.markdown(create_info_box(f"{barrier_option_type} {option_side.capitalize()} (Analytical) Price",  f"${price_analytic:.4f}"), unsafe_allow_html=True)
            with colC:
                #st.write(f"**Difference (Binomial - Analytical):** {price_binomial - price_analytic:.4e}")
                st.markdown(create_info_box(f"Difference (Binomial - Analytical)", f"{price_binomial - price_analytic:.4f}"), unsafe_allow_html=True)
        else:
            st.warning("Analytical formula returned None for these parameters.")

# Plot the binomial tree (enhanced)
col1, col2 = st.columns(2)
with col1:
    st.subheader("Binomial Tree Visualization")
    if option_style == "Vanilla":
        fig = plot_binomial_tree_enhanced(S0, sigma, T, steps, barrier_level=None)
    else:
        fig = plot_binomial_tree_enhanced(S0, sigma, T, steps, barrier_level=H)
    st.plotly_chart(fig, use_container_width=True)

# Optional Error Analysis for Barrier Options
if option_style == "Barrier":

    with col2:
        # We'll compute the closed-form once
        cf_type = combine_barrier_and_side(barrier_option_type, option_side)
        analytic_price = barrier_option_price(S0, K, T, r, q_div, sigma, H, cf_type)

        if analytic_price is None:
            st.warning("No valid analytical price for these parameters.")
        else:
            max_steps_for_error = st.slider("Max steps for error analysis", 1, 400, 10)
            data_points = []
            for n in range(1, max_steps_for_error + 1):
                b_price = barrier_binomial_option_price(
                    S0, K, r, q_div, T, sigma, n, barrier_option_type, H, option_side, rebate
                )
                err = b_price - analytic_price
                data_points.append({"Steps": n, "Binomial": b_price, "Error": err})

            df_error = pd.DataFrame(data_points)

            # Show table
            #st.dataframe(df_error)

            # Plot Binomial vs. Steps and Analytical
            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(
                x=df_error["Steps"], y=df_error["Binomial"],
                mode='lines+markers', name='Binomial Price'
            ))
            # Add a horizontal line for the analytical price
            fig_conv.add_shape(type='line',
                x0=1, x1=max_steps_for_error,
                y0=analytic_price, y1=analytic_price,
                line=dict(color='red', dash='dash'),
            )
            fig_conv.add_annotation(
                x=max_steps_for_error,
                y=analytic_price,
                xanchor='left',
                text=f"Analytical = {analytic_price:.4f}",
                showarrow=False,
                font=dict(color='red')
            )
            fig_conv.update_layout(
                title="Convergence of Binomial Price to Analytical Price",
                xaxis_title="Steps",
                yaxis_title="Price",
            )
            st.plotly_chart(fig_conv, use_container_width=True)

            # Plot the Error
            # fig_err = go.Figure()
            # fig_err.add_trace(go.Scatter(
            #     x=df_error["Steps"], y=df_error["Error"],
            #     mode='lines+markers', name='Error (Binomial - Analytical)'
            # ))
            # fig_err.update_layout(
            #     title="Error vs. Number of Steps",
            #     xaxis_title="Steps",
            #     yaxis_title="Error"
            # )
            # st.plotly_chart(fig_err, use_container_width=True)

# --- Multi-step Analysis Section ---
st.subheader("Multi-step Analysis for Different Time Steps")

# Input field for comma-separated values (e.g., "5,10,20,50,100")
steps_input = st.text_input(
    "Enter 5 values for the number of time steps (comma-separated):", 
    "5,10,20,50,100"
)

# Parse the input into a list of integers
try:
    steps_list = [int(x.strip()) for x in steps_input.split(",") if x.strip().isdigit()]
except Exception as e:
    st.error("Please enter valid integer values separated by commas.")
    steps_list = []

# Check that exactly 5 values are provided
if len(steps_list) != 5:
    st.warning("Please enter exactly 5 time step values.")
else:
    results = []
    
    if option_style == "Vanilla":
        # Compute the analytical price once using Black-Scholes.
        analytic_price = black_scholes(S0, K, T, r, sigma, option_side)
        for n in steps_list:
            bin_price = european_binomial_option_price(S0, K, r, q_div, T, sigma, n, option_side)
            error = bin_price - analytic_price
            results.append({
                "Steps": n, 
                "Binomial Price": bin_price,
                "Analytical Price": analytic_price,
                "Error": error
            })
    else:
        # For barrier options, obtain the closed-form type and price.
        cf_type = combine_barrier_and_side(barrier_option_type, option_side)
        analytic_price = barrier_option_price(S0, K, T, r, q_div, sigma, H, cf_type)
        for n in steps_list:
            bin_price = barrier_binomial_option_price(
                S0, K, r, q_div, T, sigma, n, barrier_option_type, H, option_side, rebate
            )
            error = bin_price - analytic_price if analytic_price is not None else None
            results.append({
                "Steps": n, 
                "Binomial Price": bin_price,
                "Analytical Price": analytic_price if analytic_price is not None else np.nan,
                "Error": error if error is not None else np.nan
            })
    
    df_results = pd.DataFrame(results)
    
    # --- Create a styled Plotly Table ---
    fig_table = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Steps</b>", "<b>Binomial Price</b>", "<b>Analytical Price</b>", "<b>Error</b>"],
            fill_color='paleturquoise',
            align='center',
            font=dict(size=14)
        ),
        cells=dict(
            values=[
                df_results["Steps"],
                df_results["Binomial Price"].map("{:.4f}".format),
                df_results["Analytical Price"].map("{:.4f}".format),
                df_results["Error"].map("{:.4f}".format)
            ],
            fill_color='lavender',
            align='center',
            font=dict(size=13)
        )
    )])
    
    # --- Create the Error vs. Time Steps Plot ---
    fig_error = go.Figure()
    fig_error.add_trace(go.Scatter(
        x=df_results["Steps"],
        y=df_results["Error"],
        mode='lines+markers',
        name='Error'
    ))
    fig_error.update_layout(
        title="Error vs. Time Steps",
        xaxis_title="Time Steps",
        yaxis_title="Error",
    )
    
    # --- Display Table and Plot Side by Side ---
    col_table, col_plot = st.columns(2)
    with col_table:
        st.plotly_chart(fig_table, use_container_width=True)
    with col_plot:
        st.plotly_chart(fig_error, use_container_width=True)


# Only proceed if we are dealing with a Barrier option
if option_style == "Barrier":
    # Ask user for inputs on how to vary S0 and how many steps to use
    steps_for_error = st.number_input("Number of Binomial Steps for the Error Analysis:", value=50, min_value=1)
    S0_min = st.number_input("Minimum S0", value=50.0, step=1.0)
    S0_max = st.number_input("Maximum S0", value=150.0, step=1.0)
    S0_step = st.number_input("Increment for S0", value=5.0, step=1.0)

    # Button to generate the plot
    if st.button("Plot Error vs. Stock Price"):
        # Build an array of S0 values from S0_min to S0_max in increments of S0_step
        S0_values = np.arange(S0_min, S0_max + S0_step, S0_step)

        errors = []
        # For each S0 in this range, compute binomial & analytical prices, then error
        for s in S0_values:
            bin_price = barrier_binomial_option_price(
                s, K, r, q_div, T, sigma, steps_for_error,
                barrier_option_type, H, option_side, rebate
            )
            # Convert barrier type + side to the correct analytical label
            cf_type = combine_barrier_and_side(barrier_option_type, option_side)
            an_price = barrier_option_price(s, K, T, r, q_div, sigma, H, cf_type)
            
            # If analytical formula returns None, we cannot compute an error
            if an_price is not None:
                err = bin_price - an_price
            else:
                err = np.nan  # or 0, or skip it entirely
            
            errors.append(err)

        # Create the Plotly figure for Error vs. S0
        fig_err_s0 = go.Figure()
        fig_err_s0.add_trace(go.Scatter(
            x=S0_values,
            y=errors,
            mode='lines+markers',
            name='Error (Binomial - Analytical)'
        ))
        fig_err_s0.update_layout(
            title="Error vs. Stock Price (S0)",
            xaxis_title="Stock Price (S0)",
            yaxis_title="Error",
            template="simple_white"
        )

        # Display the figure in Streamlit
        st.plotly_chart(fig_err_s0, use_container_width=True)
        
   
else:
    st.info("This section is only available for Barrier options.")



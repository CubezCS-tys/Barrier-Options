# import streamlit as st
# import math

# def european_binomial_option_price(S0, K, r, T, sigma, steps, option_type='call'):
#     """
#     Prices a European option (call or put) using the Cox-Ross-Rubinstein binomial model.
    
#     Parameters:
#     -----------
#     S0 : float
#         Current underlying asset price
#     K : float
#         Strike price of the option
#     r : float
#         Risk-free interest rate (annualized, as a decimal)
#     T : float
#         Time to maturity in years
#     sigma : float
#         Volatility of the underlying asset (annualized, as a decimal)
#     steps : int
#         Number of time steps in the binomial model
#     option_type : str
#         'call' or 'put'
    
#     Returns:
#     --------
#     float
#         The price of the option
#     """
#     # Time step
#     dt = T / steps
    
#     # Cox-Ross-Rubinstein up & down factors
#     u = math.exp(sigma * math.sqrt(dt))  # up factor
#     d = 1.0 / u                          # down factor
    
#     # Risk-neutral probability
#     R = math.exp(r * dt)         # growth per step at risk-free rate
#     q = (R - d) / (u - d)        # risk-neutral up probability
    
#     # 1) Compute asset prices at maturity for all possible paths
#     #    (the last layer of the binomial tree).
#     asset_prices = [(S0 * (u ** j) * (d ** (steps - j))) for j in range(steps + 1)]
    
#     # 2) Compute option payoffs at maturity
#     if option_type.lower() == 'call':
#         option_values = [max(0.0, price - K) for price in asset_prices]
#     else:  # put
#         option_values = [max(0.0, K - price) for price in asset_prices]
        
#     # 3) Step back through the binomial tree
#     #    Discount option values at each node to get today's price
#     for _ in range(steps):
#         for i in range(len(option_values) - 1):
#             # Expected option value under risk-neutral measure, discounted back
#             option_values[i] = (1/R) * (q * option_values[i+1] + (1 - q) * option_values[i])
#         # At each step, the list of option values gets shorter by 1
#         option_values.pop()
    
#     # The first element now contains the option price at t=0
#     return option_values[0]

# # Streamlit App
# def main():
#     st.title("European Option Pricing with the Binomial Tree Method")

#     st.markdown("""
#     This Streamlit app calculates the **European Call and Put Option** prices 
#     using the **Cox-Ross-Rubinstein (CRR) Binomial Tree** model.
#     """)

#     with st.sidebar:
#         st.header("Model Parameters")
#         S0 = st.number_input("Initial Stock Price (S0)", value=100.0, min_value=0.0, step=1.0)
#         K = st.number_input("Strike Price (K)", value=100.0, min_value=0.0, step=1.0)
#         r = st.number_input("Risk-Free Rate (r)", value=0.05, min_value=0.0, step=0.01, format="%.4f")
#         T = st.number_input("Time to Maturity (T in years)", value=1.0, min_value=0.0, step=0.25)
#         sigma = st.number_input("Volatility (sigma)", value=0.2, min_value=0.0, step=0.01, format="%.4f")
#         steps = st.number_input("Number of Steps in Binomial Tree", value=3, min_value=1, step=1)
        
#         option_type = st.selectbox("Option Type", ["call", "put"])

#     if st.button("Calculate Option Price"):
#         price = european_binomial_option_price(S0, K, r, T, sigma, steps, option_type)
#         st.write(f"**The {option_type.capitalize()} option price is:** {price:.4f}")

# if __name__ == '__main__':
#     main()


import streamlit as st
import math
import numpy as np
import plotly.graph_objects as go

# -------------------------------
# 1. Vanilla European Option Pricing
# -------------------------------
def european_binomial_option_price(S0, K, r, T, sigma, steps, option_type='call'):
    """
    Prices a European option (call or put) using the Cox-Ross-Rubinstein binomial model.
    """
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))  # up factor
    d = 1.0 / u                          # down factor
    R = math.exp(r * dt)                 # risk-free growth per step
    q = (R - d) / (u - d)                # risk-neutral probability
    
    # Terminal asset prices at maturity
    asset_prices = [S0 * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)]
    
    # Terminal payoffs
    if option_type.lower() == 'call':
        option_values = [max(price - K, 0) for price in asset_prices]
    else:
        option_values = [max(K - price, 0) for price in asset_prices]
        
    # Backward induction through the tree
    for _ in range(steps):
        for i in range(len(option_values) - 1):
            option_values[i] = (1/R) * (q * option_values[i+1] + (1 - q) * option_values[i])
        option_values.pop()
    
    return option_values[0]

# -------------------------------
# 2. Barrier Option Pricing via Binomial Tree (Recursive)
# -------------------------------
def barrier_binomial_option_price(S0, K, r, T, sigma, steps, barrier_option_type, H, option_side='call', rebate=0.0):
    """
    Prices a barrier option using a binomial tree that tracks whether the barrier has been hit.
    
    Parameters:
      - S0, K, r, T, sigma, steps: as before.
      - barrier_option_type: one of "Up-and-Out", "Down-and-Out", "Up-and-In", "Down-and-In"
      - H: barrier level
      - option_side: "call" or "put"
      - rebate: rebate paid if the option is knocked out (assumed paid at maturity)
    
    Returns:
      - Option price.
    """
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    R = math.exp(r * dt)
    q = (R - d) / (u - d)
    
    # Determine barrier direction ("up" or "down") from the barrier_option_type.
    if barrier_option_type.lower().startswith("up"):
        barrier_direction = "up"
    elif barrier_option_type.lower().startswith("down"):
        barrier_direction = "down"
    else:
        raise ValueError("Invalid barrier option type.")
        
    # Define intrinsic payoff (for European options)
    def intrinsic(S):
        if option_side.lower() == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    # Helper: update barrier-hit flag based on new asset price S_new.
    def update_bh(S_new, current_bh):
        if current_bh:
            return True
        if barrier_direction == "up":
            return S_new >= H
        else:
            return S_new <= H

    # For knock-out options, once barrier is hit the option becomes "dead".
    # We assume the rebate (if any) is paid at maturity; thus, if knocked out at time t,
    # the value is: rebate * exp(-r*(T-t)).
    is_knock_out = barrier_option_type.lower().endswith("out")
    is_knock_in  = barrier_option_type.lower().endswith("in")
    
    memo = {}
    # f(i, j, bh): value at node corresponding to step i, having taken j up moves,
    # with bh indicating whether the barrier has been hit on the path.
    def f(i, j, bh):
        key = (i, j, int(bh))
        if key in memo:
            return memo[key]
        # Current time and asset price:
        t = i * dt
        S = S0 * (u ** j) * (d ** (i - j))
        
        # Base case: at maturity
        if i == steps:
            if is_knock_out:
                # For knock-out: if barrier was hit, payoff is rebate; else, normal intrinsic payoff.
                value = rebate if bh else intrinsic(S)
            else:  # knock-in
                # For knock-in: option only exists if barrier has been hit.
                value = intrinsic(S) if bh else 0.0
            memo[key] = value
            return value
        
        # For knock-out options, if already knocked out, terminate early:
        if is_knock_out and bh:
            value = rebate * math.exp(-r * (T - t))
            memo[key] = value
            return value
        
        # Otherwise, continue to next step.
        # Compute the value for an up move:
        S_up = S0 * (u ** (j + 1)) * (d ** ((i + 1) - (j + 1)))  # = S0 * u^(j+1)*d^(i-j)
        new_bh_up = update_bh(S_up, bh)
        # Compute the value for a down move:
        S_down = S0 * (u ** j) * (d ** ((i + 1) - j))
        new_bh_down = update_bh(S_down, bh)
        
        # For knock-out options, if a move causes a barrier breach, we can terminate that branch:
        if is_knock_out and new_bh_up:
            value_up = rebate * math.exp(-r * (T - (t + dt)))
        else:
            value_up = f(i + 1, j + 1, new_bh_up)
            
        if is_knock_out and new_bh_down:
            value_down = rebate * math.exp(-r * (T - (t + dt)))
        else:
            value_down = f(i + 1, j, new_bh_down)
        
        value = (1/R) * (q * value_up + (1 - q) * value_down)
        memo[key] = value
        return value

    return f(0, 0, False)

# -------------------------------
# 3. Visualization of the Binomial Tree
# -------------------------------
def plot_binomial_tree(S0, sigma, T, steps, barrier_level=None):
    """
    Plots the asset price binomial tree. Each node is computed as S0 * u^j * d^(i-j),
    where dt = T/steps, u = exp(sigma*sqrt(dt)) and d = 1/u.
    If barrier_level is provided, a horizontal line is added.
    """
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    
    # Collect nodes and edges
    nodes = []
    edges_x = []
    edges_y = []
    
    for i in range(steps + 1):
        for j in range(i + 1):
            S = S0 * (u ** j) * (d ** (i - j))
            nodes.append((i, S))
            # Create edges from current node to next nodes (if not at final step)
            if i < steps:
                # Up move
                nodes_up = (i + 1, S0 * (u ** (j + 1)) * (d ** ((i + 1) - (j + 1))))
                edges_x.extend([i, i + 1, None])
                edges_y.extend([S, nodes_up[1], None])
                # Down move
                nodes_down = (i + 1, S0 * (u ** j) * (d ** ((i + 1) - j)))
                edges_x.extend([i, i + 1, None])
                edges_y.extend([S, nodes_down[1], None])
    
    fig = go.Figure()
    # Plot the edges (tree lines)
    fig.add_trace(go.Scatter(x=edges_x, y=edges_y, mode='lines', line=dict(color='gray'), showlegend=False))
    # Plot the nodes
    xs, ys = zip(*nodes)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers', marker=dict(size=8, color='black'), name='Nodes'))
    
    # Add barrier level line if provided
    if barrier_level is not None:
        fig.add_trace(go.Scatter(
            x=[0, steps],
            y=[barrier_level, barrier_level],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Barrier Level'
        ))
    
    fig.update_layout(
        title="Binomial Tree of Underlying Asset Prices",
        xaxis_title="Time Step",
        yaxis_title="Asset Price",
        template="simple_white"
    )
    return fig

# -------------------------------
# 4. Streamlit App Interface
# -------------------------------
def main():
    st.title("Option Pricing with Binomial Trees")
    
    st.markdown("""
    This app prices European options via the CRR binomial tree method. 
    It can also price barrier options (up-and-out, down-and-out, up-and-in, down-and-in) 
    for both calls and puts. A visualization of the underlying binomial tree is shown below.
    """)
    
    with st.sidebar:
        st.header("Model Parameters")
        S0 = st.number_input("Initial Stock Price (S0)", value=100.0, min_value=0.0, step=1.0)
        K = st.number_input("Strike Price (K)", value=100.0, min_value=0.0, step=1.0)
        r = st.number_input("Risk-Free Rate (r)", value=0.05, min_value=0.0, step=0.01, format="%.4f")
        T = st.number_input("Time to Maturity (T in years)", value=1.0, min_value=0.0, step=0.25)
        sigma = st.number_input("Volatility (sigma)", value=0.2, min_value=0.0, step=0.01, format="%.4f")
        steps = st.number_input("Number of Steps in the Binomial Tree", value=5, min_value=1, step=1)
        
        option_style = st.radio("Select Option Style", ("Vanilla", "Barrier"))
        option_side = st.selectbox("Option Side", ["call", "put"])
    
        if option_style == "Vanilla":
            st.info("Pricing a standard European option using the binomial tree.")
        else:
            barrier_option_type = st.selectbox("Barrier Option Type", 
                                                 ["Up-and-Out", "Down-and-Out", "Up-and-In", "Down-and-In"])
            H = st.number_input("Barrier Level (H)", value=110.0, min_value=0.0, step=1.0)
            rebate = st.number_input("Rebate (if knocked out)", value=0.0, min_value=0.0, step=0.1)
    
    if st.button("Calculate Option Price"):
        if option_style == "Vanilla":
            price = european_binomial_option_price(S0, K, r, T, sigma, steps, option_side)
            st.write(f"**Vanilla {option_side.capitalize()} Option Price:** {price:.4f}")
        else:
            price = barrier_binomial_option_price(S0, K, r, T, sigma, steps, barrier_option_type, H, option_side, rebate)
            st.write(f"**{barrier_option_type} {option_side.capitalize()} Barrier Option Price:** {price:.4f}")
    
    st.subheader("Binomial Tree Visualization")
    if option_style == "Vanilla":
        fig = plot_binomial_tree(S0, sigma, T, steps)
    else:
        fig = plot_binomial_tree(S0, sigma, T, steps, barrier_level=H)
    st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()

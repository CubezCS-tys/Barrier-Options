
# import streamlit as st

# st.write("""
# ### Regular Call and Put Options
# The prices at time zero of a regular European call option ($c$) and put option ($p$) are given by:
# """)
# st.latex(r"""
# c = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)
# """)
# st.latex(r"""
# p = K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)
# """)
# st.write("Where:")
# st.latex(r"""
# d_1 = \frac{\ln(S_0/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad
# d_2 = d_1 - \sigma\sqrt{T}
# """)

# st.write("""
# ### Down-and-In Call Option for $H$ $\leq$ $K$
# """)
# st.latex(r"""
# c_{di} = S_0 e^{-qT} \left(\frac{H}{S_0}\right)^{2\lambda} N(y) - K e^{-rT} \left(\frac{H}{S_0}\right)^{2\lambda-2} N(y - \sigma\sqrt{T})
# """)
# st.latex(r"""
# \text{Where: } 
# \lambda = \frac{r - q + \sigma^2/2}{\sigma^2}, \quad
# y = \frac{\ln(H^2 / (S_0 K))}{\sigma\sqrt{T}} + \lambda\sigma\sqrt{T}
# """)

# st.write("""
# ### Down-and-Out Call Option for $H$ $\leq$ $K$
# """)
# st.latex(r"""
# c_{do} = c - c_{di}
# """)

# st.write("""
# ### Down-and-Out Call Option for $H$ $\geq$ $K$
# """)

# st.latex(r"""
# c_{do} = S_0 N(x_1) e^{-qT} - K e^{-rT} N(x_1 - \sigma\sqrt{T}) 
#         - S_0 e^{-qT} \left(\frac{H}{S_0}\right)^{2\lambda} N(y_1) 
#         + K e^{-rT} \left(\frac{H}{S_0}\right)^{2\lambda-2} N(y_1 - \sigma\sqrt{T})
# """)
# st.latex(r"""
# \text{Where: }
# x_1 = \frac{\ln(S_0 / H)}{\sigma\sqrt{T}} + \lambda\sigma\sqrt{T}, \quad
# y_1 = \frac{\ln(H / S_0)}{\sigma\sqrt{T}} + \lambda\sigma\sqrt{T}
# """)

# st.write("""
# ### Down-and-In Call Option for $H$ $\geq$ $K$
# """)

# st.latex(r"""
# c_{di} = c - c_{do}
# """)


# st.write("""
# ### Up-and-In Call Option for $H$ $\geq$ $K$

# """)
# st.latex(r"""
# c_{ui} = S_0 N(x_1) e^{-qT} - K e^{-rT} N(x_1 - \sigma\sqrt{T}) 
#         - S_0 e^{-qT} \left(\frac{H}{S_0}\right)^{2\lambda}[N(-y) - N(-y_1)] + K e^{-rT} \left(\frac{H}{S_0}\right)^{2\lambda-2} [N(-y + \sigma\sqrt{T} - N(-y_1 + \sigma\sqrt{T})]
# """)

# st.write("""
# ### Up-and-Out Call Option for $H$ $\geq$ $K$
# """)
# st.latex(r"""
# c_{uo} = c - c_{ui}
# """)

# st.write(""" 
#         ### Up-and-In Put Option for $H$ $\geq$ $K$
#         """)
# st.latex(r"""
# p_{ui} = -S_0 e^{-qT} \left(\frac{H}{S_0}\right)^{2\lambda} N(-y) 
#         + K e^{-rT} \left(\frac{H}{S_0}\right)^{2\lambda-2} N(-y + \sigma\sqrt{T})
# """)

# st.write(""" 
#         ### Up-and-Out Put Option for $H$ $\geq$ $K$
#         """)

# st.latex(r"""
# p_{uo} = p - p_{ui}
# """)

# st.write(""" 
#         ### Up-and-Out Put Option for $H$ $\leq$ $K$
#         """)
# st.latex(r"""
#     p_{uo} = -S_0 N(-x_1) e^{-qT} + K e^{-rT} N(-x_1 + \sigma\sqrt{T}) 
#         + S_0 e^{-qT} \left(\frac{H}{S_0}\right)^{2\lambda} N(-y_1) 
#         - K e^{-rT} \left(\frac{H}{S_0}\right)^{2\lambda-2} N(-y_1 + \sigma\sqrt{T})
#         """)

# st.write(""" 
#         ### Up-and-In Put Option for $H$ $\leq$ $K$
#         """)
# st.latex(r"""
# p_{ui} = p - p_{uo}
# """)

# st.write(""" 
#         ### Down-and-Out Put Option for $H$ $\geq$ $K$
#         """)
# st.latex(r"""
# p_{do} = 0
# """)

# st.write(""" 
#         ### Down-and-In Put Option for $H$ $\geq$ $K$
#         """)
# st.latex(r"""
# p_{di} = p
# """)

# st.write(""" 
#         ### Down-and-In Put Option for $H$ $\leq$ $K$
#         """)

# st.latex(r"""
#     p_{uo} = -S_0 N(-x_1) e^{-qT} + K e^{-rT} N(-x_1 + \sigma\sqrt{T}) 
#         + S_0 e^{-qT} \left(\frac{H}{S_0}\right)^{2\lambda} [N(y) - N(y_1)] - K e^{-rT} \left(\frac{H}{S_0}\right)^{2\lambda - 2} [N(y - \sigma\sqrt{T}) - N(y_1 - \sigma\sqrt{T})]
#         """)

# st.write(""" 
#         ### Down-and-Out Put Option for $H$ $\leq$ $K$
#         """)
# st.latex(r"""
# p_{do} = p - p_{di}
# """)

import streamlit as st
import numpy as np
import math
import time
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.interpolate import interp1d

# =============================================================================
# Functions from Binomial_Adaptive_Mesh.py
# =============================================================================

def calc_d1(S0, K, r, q, sigma, T):
    return (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def calc_d2(S0, K, r, q, sigma, T):
    return calc_d1(S0, K, r, q, sigma, T) - sigma * np.sqrt(T)

def calc_c(S0, K, r, q, sigma, T):
    """Analytical price of a plain vanilla Call via Black-Scholes."""
    d1 = calc_d1(S0, K, r, q, sigma, T)
    d2 = calc_d2(S0, K, r, q, sigma, T)
    return (S0 * np.exp(-q*T) * norm.cdf(d1)
            - K * np.exp(-r*T) * norm.cdf(d2))

def calc_p(S0, K, r, q, sigma, T):
    """Analytical price of a plain vanilla Put via Black-Scholes."""
    d1 = calc_d1(S0, K, r, q, sigma, T)
    d2 = calc_d2(S0, K, r, q, sigma, T)
    return (K * np.exp(-r*T) * norm.cdf(-d2)
            - S0 * np.exp(-q*T) * norm.cdf(-d1))

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
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def barrier_option_price(S0, K, T, r, q, sigma, barrier, option_type):
    """
    Analytical price of various knock-in/out barrier options.
    option_type examples: 'down-and-in call', 'down-and-out call',
                          'up-and-in call', 'up-and-out call', etc.
    """
    x1 = calc_x1(S0, barrier, T, sigma, r, q)
    y1 = calc_y1(S0, barrier, T, sigma, r, q)
    c = calc_c(S0, K, r, q, sigma, T)
    p = calc_p(S0, K, r, q, sigma, T)
    lam = calc_lambda(r, q, sigma)
    y = calc_y(barrier, S0, K, T, sigma, r, q)

    # Down-and-in Call
    if option_type == 'down-and-in call' and barrier <= K and S0 <= barrier:
        vanilla = black_scholes(S0, K, T, r, sigma, "call")
        return vanilla
    elif option_type == 'down-and-in call' and barrier <= K:
        cdi = (S0 * np.exp(-q*T) * (barrier/S0)**(2*lam) * norm.cdf(y)
               - K * np.exp(-r*T) * (barrier/S0)**(2*lam-2)
                 * norm.cdf(y - sigma*np.sqrt(T)))
        return cdi
    elif option_type == 'down-and-in call' and barrier >= K:
        term1 = S0*np.exp(-q*T)*norm.cdf(x1)
        term2 = K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
        term3 = S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(y1)
        term4 = K*np.exp(-r*T)*(barrier/S0)**(2*lam-2)*norm.cdf(y1 - sigma*np.sqrt(T))
        cdo = term1 - term2 - term3 + term4
        if cdo < 0: cdo = 0
        cdi = c - cdo
        return cdi

    # Down-and-out Call
    elif option_type == 'down-and-out call' and barrier <= K:
        cdi = (S0 * np.exp(-q*T) * (barrier/S0)**(2*lam) * norm.cdf(y)
               - K * np.exp(-r*T) * (barrier/S0)**(2*lam-2)
                 * norm.cdf(y - sigma*np.sqrt(T)))
        cdo = c - cdi
        return max(cdo, 0)
    elif option_type == 'down-and-out call' and barrier >= K:
        term1 = S0 * np.exp(-q*T)*norm.cdf(x1)
        term2 = K  * np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
        term3 = S0 * np.exp(-q*T)*((barrier/S0)**(2*lam))*norm.cdf(y1)
        term4 = K  * np.exp(-r*T)*((barrier/S0)**(2*lam-2))*norm.cdf(y1 - sigma*np.sqrt(T))
        cdo = term1 - term2 - term3 + term4
        return max(cdo, 0)

    # Up-and-in Call
    elif option_type == 'up-and-in call' and barrier > K:
        cui = (S0*np.exp(-q*T)*norm.cdf(x1)
               - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
               - S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*(norm.cdf(-y)-norm.cdf(-y1))
               + K*np.exp(-r*T)*(barrier/S0)**(2*lam-2)
                 *(norm.cdf(-y+sigma*np.sqrt(T))-norm.cdf(-y1+sigma*np.sqrt(T))))
        return cui
    elif option_type == 'up-and-in call' and barrier <= K:
        return c

    # Up-and-out Call
    elif option_type == 'up-and-out call' and barrier <= K:
        return 0.0
    elif option_type == 'up-and-out call' and barrier > K:
        cui = (S0*np.exp(-q*T)*norm.cdf(x1)
               - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
               - S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*(norm.cdf(-y)-norm.cdf(-y1))
               + K*np.exp(-r*T)*(barrier/S0)**(2*lam-2)
                 *(norm.cdf(-y+sigma*np.sqrt(T))-norm.cdf(-y1+sigma*np.sqrt(T))))
        cuo = c - cui
        return max(cuo, 0)

    # Up-and-in Put
    elif option_type == 'up-and-in put' and barrier >= K and barrier <= S0:
        pui = black_scholes(S0, K, T, r, sigma, "put")
        return pui
    elif option_type == 'up-and-in put' and barrier >= K:
        pui = (-S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(-y)
               + K*np.exp(-r*T)*(barrier/S0)**(2*lam-2)*norm.cdf(-y+sigma*np.sqrt(T)))
        return pui
    elif option_type == 'up-and-in put' and barrier <= K:
        return p

    # Up-and-out Put
    elif option_type == 'up-and-out put' and barrier >= K:
        pui = (-S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(-y)
               + K*np.exp(-r*T)*(barrier/S0)**(2*lam-2)*norm.cdf(-y+sigma*np.sqrt(T)))
        puo = p - pui
        return max(puo, 0)
    elif option_type == 'up-and-out put' and barrier <= K:
        puo = (-S0*np.exp(-q*T)*norm.cdf(-x1)
               + K*np.exp(-r*T)*norm.cdf(-x1+sigma*np.sqrt(T))
               + S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(-y1)
               - K*np.exp(-r*T)*(barrier/S0)**(2*lam-2)*norm.cdf(-y1+sigma*np.sqrt(T)))
        return max(puo, 0)

    # Down-and-in Put
    elif option_type == 'down-and-in put' and barrier < K and S0 < barrier:
        vanilla = black_scholes(S0, K, T, r, sigma, "put")
        return vanilla
    elif option_type == 'down-and-in put' and barrier > K:
        return p
    elif option_type == 'down-and-in put' and barrier < K:
        pdi = (-S0*np.exp(-q*T)*norm.cdf(-x1)
               + K*np.exp(-r*T)*norm.cdf(-x1+sigma*np.sqrt(T))
               + S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*(norm.cdf(y)-norm.cdf(y1))
               - K*np.exp(-r*T)*(barrier/S0)**(2*lam-2)
                 *(norm.cdf(y-sigma*np.sqrt(T))-norm.cdf(y1-sigma*np.sqrt(T)))
              )
        return pdi

    # Down-and-out Put
    elif option_type == 'down-and-out put' and barrier > K:
        return 0
    elif option_type == 'down-and-out put' and barrier < K:
        pdi = (-S0*np.exp(-q*T)*norm.cdf(-x1)
               + K*np.exp(-r*T)*norm.cdf(-x1+sigma*np.sqrt(T))
               + S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*(norm.cdf(y)-norm.cdf(y1))
               - K*np.exp(-r*T)*(barrier/S0)**(2*lam-2)
                 *(norm.cdf(y-sigma*np.sqrt(T))-norm.cdf(y1-sigma*np.sqrt(T)))
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

        S_up = S0 * (u ** (j+1)) * (d ** ((i+1) - (j+1)))
        S_down = S0 * (u**j) * (d ** ((i+1) - j))

        def barrier_hit(S_new, current_bh):
            if current_bh:
                return True
            if barrier_direction == "up":
                return S_new >= H
            else:
                return S_new <= H

        new_bh_up = barrier_hit(S_up, bh)
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
        key = (round(S,4), round(t,6), barrier_hit)
        if key in memo:
            return memo[key]

        val_up = adaptive_f(S_up, t + dt, barrier_hit)
        val_down = adaptive_f(S_down, t + dt, barrier_hit)
        val = disc * (p * val_up + (1 - p) * val_down)
        memo[key] = val
        return val

    return adaptive_f(S0, 0, False)

def combine_barrier_and_side(barrier_option_type, side):
    return f"{barrier_option_type.lower().replace(' ', '-')} {side.lower()}"

# =============================================================================
# Functions from Binomial_Adaptive_Plot.py
# =============================================================================

def plot_adaptive_binomial_tree(S0, sigma, T, coarse_steps, fine_steps, barrier, fine_region):
    critical_region = (barrier * (1 - fine_region), barrier * (1 + fine_region))
    dt_coarse = T / coarse_steps if coarse_steps > 0 else T
    dt_fine = T / fine_steps if fine_steps > 0 else T

    nodes = []
    edges_x, edges_y = [], []

    def add_node(S, t, parent_x=None, parent_y=None):
        if t > T:
            return
        nodes.append((t, S))
        if parent_x is not None:
            edges_x.extend([parent_x, t, None])
            edges_y.extend([parent_y, S, None])
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

# =============================================================================
# Merged Streamlit App with Tabs for Pricing and Tree Visualization
# =============================================================================

st.set_page_config(page_title="Binomial Adaptive Options", layout="wide")

tab1, tab2 = st.tabs(["Pricing", "Tree Visualization"])

with tab1:
    st.title("Adaptive Mesh Refinement for Barrier Options (Binomial Model)")
    col1, col2 = st.columns(2)
    with col1:
        st.header("Model Parameters")
        S0_input = st.number_input("Initial Price S0", value=100.0, min_value=0.0)
        K_input = st.number_input("Strike Price K", value=100.0, min_value=0.0)
        r_input = st.number_input("Risk-free Rate r", value=0.05, min_value=0.0)
        q_input = st.number_input("Dividend Yield q", value=0.0, min_value=0.0)
        T_input = st.number_input("Time to Maturity T", value=1.0, min_value=0.0)
        sigma_input = st.number_input("Volatility σ", value=0.2, min_value=0.0)
        barrier_input = st.number_input("Barrier Level", value=110.0, min_value=0.0)
        option_side = st.selectbox("Option Side", ["call", "put"])
        barrier_type = st.selectbox("Barrier Type", ["Up-and-Out", "Down-and-Out", "Up-and-In", "Down-and-In"])
    with col2:
        st.header("AMR Parameters")
        coarse_steps = st.number_input("Coarse Steps", value=20, min_value=1)
        fine_steps = st.number_input("Fine Steps (around barrier)", value=400, min_value=1)
        fine_region = st.slider("Fine Region Width (%) around Barrier", 0.01, 0.3, 0.1)
    
    if st.button("Calculate Pricing"):
        start = time.time()
        adaptive_price = adaptive_barrier_binomial(S0_input, K_input, r_input, q_input, T_input, sigma_input,
                                                   coarse_steps, fine_steps, barrier_input,
                                                   barrier_type.lower(), option_side, 0.0, fine_region)
        adaptive_time = time.time() - start

        start = time.time()
        regular_price = barrier_binomial_option_price(S0_input, K_input, r_input, q_input, T_input, sigma_input,
                                                      coarse_steps, barrier_type.lower(), barrier_input,
                                                      option_side, 0.0)
        regular_time = time.time() - start

        cf_type = combine_barrier_and_side(barrier_type, option_side)
        an_price = barrier_option_price(S0_input, K_input, T_input, r_input, q_input, sigma_input,
                                        barrier_input, cf_type)
        st.subheader("Results Comparison")
        st.write(f"Adaptive Binomial Price: **{adaptive_price:.4f}** (Time: {adaptive_time:.4f}s)")
        st.write(f"Regular Binomial Price: **{regular_price:.4f}** (Time: {regular_time:.4f}s)")
        st.write(f"Analytical Price: **{an_price:.4f}**")
        st.subheader("Interpretation")
        st.markdown("""
        - **Adaptive method** should yield more accurate results near barriers.
        - Compare computational times and errors.
        - Test various mesh refinements to balance efficiency and accuracy.
        """)

    st.subheader("Error vs. Stock Price (S0)")
    S0_min = st.number_input("Minimum S0", value=50.0, step=1.0, key='min_s0')
    S0_max = st.number_input("Maximum S0", value=150.0, step=1.0, key='max_s0')
    S0_step = st.number_input("Increment for S0", value=5.0, step=1.0, key='step_s0')
    if st.button("Generate Error Plot", key="error_plot"):
        S0_values = np.arange(S0_min, S0_max + S0_step, S0_step)
        adaptive_errors = []
        regular_errors = []
        for s in S0_values:
            a_price = adaptive_barrier_binomial(s, K_input, r_input, q_input, T_input, sigma_input,
                                                coarse_steps, fine_steps, barrier_input,
                                                barrier_type.lower(), option_side, 0.0, fine_region)
            r_price = barrier_binomial_option_price(s, K_input, r_input, q_input, T_input, sigma_input,
                                                    coarse_steps, barrier_type.lower(), barrier_input,
                                                    option_side, 0.0)
            cf_type = combine_barrier_and_side(barrier_type, option_side)
            analytic_price = barrier_option_price(s, K_input, T_input, r_input, q_input, sigma_input,
                                                  barrier_input, cf_type)
            if analytic_price is not None:
                adaptive_err = np.abs(a_price - analytic_price)
                regular_err = np.abs(r_price - analytic_price)
            else:
                adaptive_err = np.nan
                regular_err = np.nan
            adaptive_errors.append(adaptive_err)
            regular_errors.append(regular_err)
        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(
            x=S0_values, y=adaptive_errors,
            mode='lines+markers',
            line=dict(shape='linear', width=2),
            marker=dict(size=6),
            name='Adaptive Binomial'
        ))
        fig_err.add_trace(go.Scatter(
            x=S0_values, y=_

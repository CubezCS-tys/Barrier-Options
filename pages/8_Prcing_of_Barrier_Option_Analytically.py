

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# ------------------------------
# Barrier Option Pricing Functions
# ------------------------------

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

    return (r - q + 0.5 * sigma**2) / (sigma**2)

def calc_y(barrier, S0, K, T, sigma, r, q):

    lam = calc_lambda(r, q, sigma)
    return (np.log((barrier**2)/(S0*K)) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

def calc_x1(S0, barrier, T, sigma, r, q):

    lam = calc_lambda(r, q, sigma)
    return (np.log(S0/barrier) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

def calc_y1(S0, barrier, T, sigma, r, q):

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



# ------------------------------
# Streamlit UI
# ------------------------------

st.title("Analytical Barrier Option Pricing Dashboard")
st.sidebar.header("Input Parameters")

S0 = st.sidebar.number_input("Stock Price (S0)", value=100.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0)
T = st.sidebar.number_input("Term (Years)", value=1.0)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05)
q = st.sidebar.number_input("Dividend Yield (q)", value=0.03)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2)
H = st.sidebar.number_input("Barrier (H)", value=80.0)
rebate = st.sidebar.number_input("Rebate", value=0.0)

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
        "up-and-out put"
    ]
)

# Compute the analytical barrier option price
analytical_price = barrier_option_price(S0, K, T, r, q, sigma, H, option_type)

# ------------------------------
# Custom CSS for Info Boxes
# ------------------------------
st.markdown(
    """
    <style>
    .info-box {
        background-color: #f1f1f1;
        border-left: 5px solid #007ACC;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
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
    """,
    unsafe_allow_html=True
)

def create_info_box(title, value):
    return f"<div class='info-box'><h4>{title}</h4><p>{value}</p></div>"

# ------------------------------
# Display Key Output in Info Box (only the analytical barrier option price)
# ------------------------------
if analytical_price is not None:
    st.markdown(create_info_box("Analytical Barrier Option Price", f"${analytical_price:.4f}"), unsafe_allow_html=True)
else:
    st.markdown(create_info_box("Error", "Invalid option type or parameters."), unsafe_allow_html=True)

# ------------------------------
# Sensitivity Analysis: Option Price vs. Barrier Level (2D Plot)
# ------------------------------
st.subheader("Sensitivity Analysis: Option Price vs. Barrier Level")
barrier_range = np.linspace(0.8 * H, 1.2 * H, 50)
price_vs_barrier = []
for h in barrier_range:
    p_val = barrier_option_price(S0, K, T, r, q, sigma, h, option_type)
    price_vs_barrier.append(p_val if p_val is not None else np.nan)

fig_barrier = go.Figure()
fig_barrier.add_trace(go.Scatter(x=barrier_range, y=price_vs_barrier,
                                 mode='lines+markers', name="Option Price"))
fig_barrier.update_layout(
    title="Option Price vs. Barrier Level",
    xaxis_title="Barrier Level",
    yaxis_title="Option Price"
)
st.plotly_chart(fig_barrier, use_container_width=True)

# ------------------------------
# New Section: Option Price vs. Stock Price & 3D Surface Plot
# ------------------------------
st.subheader("Option Price vs. Stock Price and 3D Surface")

# 2D Plot: Option Price vs. Stock Price (today's value, with fixed barrier H)
stock_range = np.linspace(0.8 * S0, 1.2 * S0, 50)
price_vs_stock = []
for s in stock_range:
    p_val = barrier_option_price(s, K, T, r, q, sigma, H, option_type)
    price_vs_stock.append(p_val if p_val is not None else np.nan)

fig_stock = go.Figure()
fig_stock.add_trace(go.Scatter(x=stock_range, y=price_vs_stock,
                              mode='lines+markers', name="Option Price"))
fig_stock.update_layout(
    title="Option Price vs. Stock Price (Today's Value)",
    xaxis_title="Stock Price",
    yaxis_title="Option Price"
)

# 3D Surface Plot: Option Price as function of Stock Price and Barrier Level
stock_range_3d = np.linspace(0.8 * S0, 1.2 * S0, 30)
barrier_range_3d = np.linspace(0.8 * H, 1.2 * H, 30)
S_mesh, H_mesh = np.meshgrid(stock_range_3d, barrier_range_3d)
price_surface = np.zeros_like(S_mesh)
for i in range(S_mesh.shape[0]):
    for j in range(S_mesh.shape[1]):
        p_val = barrier_option_price(S_mesh[i, j], K, T, r, q, sigma, H_mesh[i, j], option_type)
        price_surface[i, j] = p_val if p_val is not None else np.nan

fig_3d = go.Figure(data=[go.Surface(x=S_mesh, y=H_mesh, z=price_surface)])
fig_3d.update_layout(
    title="3D Surface: Option Price vs. Stock Price & Barrier Level",
    scene=dict(
        xaxis_title="Stock Price",
        yaxis_title="Barrier Level",
        zaxis_title="Option Price"
    )
)

# Display the 2D and 3D plots side by side
colA, colB = st.columns(2)
with colA:
    st.plotly_chart(fig_stock, use_container_width=True)
with colB:
    st.plotly_chart(fig_3d, use_container_width=True)


# # import streamlit as st 
# # import numpy as np
# # from scipy.stats import norm

# # # Function implementing the Black-Scholes formula
# # def black_scholes(S, K, T, r, sigma):
# #     """
# #     Black-Scholes formula for European Call and Put options.
# #     S: Spot price
# #     K: Strike price
# #     T: Time to maturity
# #     r: Risk-free interest rate
# #     sigma: Volatility (standard deviation of returns)
# #     """
# #     d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
# #     d2 = d1 - sigma * np.sqrt(T)
    
# #     call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
# #     put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
# #     return call_price, put_price

# # # Streamlit page layout
# # st.set_page_config(page_title="Black-Scholes Option Pricing U")

# # st.title("Black-Scholes Option Pricing")

# # # Sidebar inputs for model parameters
# # st.sidebar.header("Input Parameters")

# # S = st.sidebar.number_input("Spot Price (S)", min_value=0.0, value=50.0)
# # K = st.sidebar.number_input("Strike Price (K)", min_value=0.0, value=50.0)
# # T = st.sidebar.number_input("Time to Maturity (T, in years)", min_value=0.01, value=1.0, step=0.01)
# # r = st.sidebar.number_input("Risk-Free Interest Rate (r)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
# # sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.3, step=0.01)

# # # Calculate option prices
# # call_price, put_price = black_scholes(S, K, T, r, sigma)

# # # CSS styles for the rounded rectangles
# # rounded_box_css = """
# # <style>
# # .box-container {
# #     display: flex;
# #     justify-content: center;
# #     gap: 40px; /* Adjusted gap for better spacing */
# #     margin-top: 30px;
# #     flex-wrap: wrap;
# # }

# # .rounded-box {
# #     background-color: #f9f9f9;
# #     padding: 25px; /* Adjusted padding */
# #     border-radius: 15px; /* Rounded corners */
# #     border: 2px solid #ddd;
# #     width:550px; /* Adjusted width */
# #     text-align: center;
# #     box-sizing: border-box;
# #     display: flex;
# #     flex-direction: column;
# #     align-items: center;
# #     justify-content: center;
# #     box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
# # }

# # .call-box {
# #     background-color: #d4edda;  /* Light green */
# #     border-color: #28a745;      /* Green border */
# # }

# # .put-box {
# #     background-color: #f8d7da;  /* Light red */
# #     border-color: #dc3545;      /* Red border */
# # }

# # .rounded-box h3 {
# #     margin: 0;
# #     font-weight: bold;
# #     font-size: 25px; /* Increased font size */
# #     color: #333;
# # }

# # .rounded-box h1 {
# #     margin: 5px 0 0 0;
# #     font-size: 28px; /* Increased font size */
# #     color: #333;
# # }
# # </style>
# # """

# # # Inject CSS styles
# # st.markdown(rounded_box_css, unsafe_allow_html=True)

# # # Display results with custom styled boxes
# # st.markdown(
# #     f"""
# #     <div class="box-container">
# #         <div class="rounded-box call-box">
# #             <h3>CALL Value</h3>
# #             <h1>${call_price:.2f}</h1>
# #         </div>
# #         <div class="rounded-box put-box">
# #             <h3>PUT Value</h3>
# #             <h1>${put_price:.2f}</h1>
# #         </div>
# #     </div>
# #     """,
# #     unsafe_allow_html=True
# # )

# # # Space between option prices and formula details
# # st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)

# # # Optional: Display formula details for context
# # with st.expander("See Black-Scholes Formula Details"):
# #     st.latex(r"""
# #     d_1 = \frac{\ln(S / K) + (r + 0.5 \sigma^2) T}{\sigma \sqrt{T}}
# #     """)
# #     st.latex(r"""
# #     d_2 = d_1 - \sigma \sqrt{T}
# #     """)
# #     st.latex(r"""
# #     \text{Call Price} = S N(d_1) - K e^{-r T} N(d_2)
# #     """)
# #     st.latex(r"""
# #     \text{Put Price} = K e^{-r T} N(-d_2) - S N(-d_1)
# #     """)

# import streamlit as st
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# from scipy.stats import norm

# # -------------------------
# # Black-Scholes & Greeks
# # -------------------------
# def black_scholes(S, K, T, r, sigma):
#     """
#     Black-Scholes formula for European Call and Put options.
#     Returns call_price, put_price, d1, d2.
#     """
#     d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)

#     call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
#     put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
#     return call_price, put_price, d1, d2

# def greeks(S, K, T, r, sigma, d1, d2):
#     """
#     Computes the main Greeks for European Call/Put under Black-Scholes:
#     Delta, Gamma, Vega, Theta, and Rho.
#     """
#     # Delta
#     call_delta = norm.cdf(d1)
#     put_delta = norm.cdf(d1) - 1.0
    
#     # Gamma
#     gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
#     # Vega
#     vega = S * np.sqrt(T) * norm.pdf(d1)
    
#     # Theta
#     call_theta = (-S * sigma * norm.pdf(d1) / (2.0 * np.sqrt(T))
#                   - r * K * np.exp(-r * T) * norm.cdf(d2))
#     put_theta = (-S * sigma * norm.pdf(d1) / (2.0 * np.sqrt(T))
#                  + r * K * np.exp(-r * T) * norm.cdf(-d2))
    
#     # Rho
#     call_rho = K * T * np.exp(-r * T) * norm.cdf(d2)
#     put_rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
#     return {
#         "Call Delta": call_delta,
#         "Put Delta": put_delta,
#         "Gamma": gamma,
#         "Vega": vega,
#         "Call Theta": call_theta,
#         "Put Theta": put_theta,
#         "Call Rho": call_rho,
#         "Put Rho": put_rho
#     }

# # -------------------------
# # Streamlit App
# # -------------------------
# st.set_page_config(
#     page_title="Black-Scholes Option Pricing Dashboard",
#     layout="wide"  # Use the entire width of the browser
# )

# st.title("Black-Scholes Option Pricing Dashboard")

# # Sidebar for user inputs
# st.sidebar.header("Input Parameters")
# S = st.sidebar.number_input("Spot Price (S)", min_value=0.0, value=50.0)
# K = st.sidebar.number_input("Strike Price (K)", min_value=0.0, value=50.0)
# T = st.sidebar.number_input("Time to Maturity (T, in years)", min_value=0.01, value=1.0, step=0.01)
# r = st.sidebar.number_input("Risk-Free Interest Rate (r)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
# sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.3, step=0.01)

# # Calculate option prices and Greeks
# call_price, put_price, d1, d2 = black_scholes(S, K, T, r, sigma)
# greeks_dict = greeks(S, K, T, r, sigma, d1, d2)

# # -------------------------
# # Top row: Call and Put values side by side
# # -------------------------
# top_col1, top_col2 = st.columns(2)

# with top_col1:
#     # Call Value Box
#     st.markdown(
#         f"""
#         <div style="
#             background-color:#d4edda;
#             border:2px solid #28a745;
#             border-radius:15px;
#             padding:20px;
#             text-align:center;
#             box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
#         ">
#         <h3 style="margin:0;">CALL Value</h3>
#         <h1 style="margin:5px 0 0 0;">${call_price:.2f}</h1>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

# with top_col2:
#     # Put Value Box
#     st.markdown(
#         f"""
#         <div style="
#             background-color:#f8d7da;
#             border:2px solid #dc3545;
#             border-radius:15px;
#             padding:20px;
#             text-align:center;
#             box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
#         ">
#         <h3 style="margin:0;">PUT Value</h3>
#         <h1 style="margin:5px 0 0 0;">${put_price:.2f}</h1>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

# # -------------------------
# # Bottom row: Payoff chart & Greeks table side by side
# # -------------------------
# bottom_col1, bottom_col2 = st.columns(2)

# with bottom_col1:
#     st.subheader("Option Payoff Functions at Expiration")
#     # Range for underlying price
#     S_range = np.linspace(0.5 * K, 1.5 * K, 100)
#     call_payoff = np.maximum(S_range - K, 0)
#     put_payoff = np.maximum(K - S_range, 0)
    
#     # Plotly payoff chart
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=S_range, y=call_payoff,
#         mode='lines', name='Call Payoff',
#         line=dict(color='green')
#     ))
#     fig.add_trace(go.Scatter(
#         x=S_range, y=put_payoff,
#         mode='lines', name='Put Payoff',
#         line=dict(color='red')
#     ))
    
#     fig.update_layout(
#         title="Option Payoff Functions",
#         xaxis_title="Underlying Price at Expiration",
#         yaxis_title="Payoff",
#         margin=dict(l=0, r=0, t=50, b=0),
#         height=450
#     )
    
#     st.plotly_chart(fig, use_container_width=True)

# with bottom_col2:
#     st.subheader("Option Greeks")
#     greeks_df = pd.DataFrame({
#         "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
#         "Call": [
#             greeks_dict["Call Delta"],
#             greeks_dict["Gamma"],
#             greeks_dict["Vega"],
#             greeks_dict["Call Theta"],
#             greeks_dict["Call Rho"]
#         ],
#         "Put": [
#             greeks_dict["Put Delta"],
#             greeks_dict["Gamma"],
#             greeks_dict["Vega"],
#             greeks_dict["Put Theta"],
#             greeks_dict["Put Rho"]
#         ]
#     })
#     st.table(greeks_df)

# # -------------------------
# # Expandable Black-Scholes details
# # -------------------------
# with st.expander("See Black-Scholes Formula Details"):
#     st.latex(r"""
#     d_1 = \frac{\ln(S / K) + (r + 0.5 \sigma^2) T}{\sigma \sqrt{T}}
#     """)
#     st.latex(r"""
#     d_2 = d_1 - \sigma \sqrt{T}
#     """)
#     st.latex(r"""
#     \text{Call Price} = S\,N(d_1) - K\,e^{-rT}\,N(d_2)
#     """)
#     st.latex(r"""
#     \text{Put Price} = K\,e^{-rT}\,N(-d_2) - S\,N(-d_1)
#     """)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call_price, put_price, d1, d2

def greeks(S, K, T, r, sigma, d1, d2):
    call_delta = norm.cdf(d1)
    put_delta = norm.cdf(d1) - 1.0
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * norm.pdf(d1)
    call_theta = (
        -S * sigma * norm.pdf(d1) / (2.0 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2)
    )
    put_theta = (
        -S * sigma * norm.pdf(d1) / (2.0 * np.sqrt(T))
        + r * K * np.exp(-r * T) * norm.cdf(-d2)
    )
    call_rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    put_rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    return {
        "Call Delta": call_delta,
        "Put Delta": put_delta,
        "Gamma": gamma,
        "Vega": vega,
        "Call Theta": call_theta,
        "Put Theta": put_theta,
        "Call Rho": call_rho,
        "Put Rho": put_rho
    }

st.set_page_config(page_title="Black-Scholes Option Pricing Dashboard", layout="wide")
st.title("Black-Scholes Option Pricing Dashboard")

# -------------------------
# Sidebar for user inputs
# -------------------------
st.sidebar.header("Input Parameters")
S = st.sidebar.number_input("Spot Price (S)", min_value=0.0, value=50.0)
K = st.sidebar.number_input("Strike Price (K)", min_value=0.0, value=50.0)
T = st.sidebar.number_input("Time to Maturity (T, in years)", min_value=0.01, value=1.0, step=0.01)
r = st.sidebar.number_input("Risk-Free Interest Rate (r)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.3, step=0.01)

# Calculate option prices and Greeks at the given S
call_price, put_price, d1, d2 = black_scholes(S, K, T, r, sigma)
greeks_dict = greeks(S, K, T, r, sigma, d1, d2)

# -------------------------
# Top row: Call and Put values
# -------------------------
top_col1, top_col2 = st.columns(2)

with top_col1:
    # Call Value
    st.markdown(
        f"""
        <div style="
            background-color:#d4edda;
            border:2px solid #28a745;
            border-radius:15px;
            padding:20px;
            text-align:center;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        ">
        <h3 style="margin:0;">CALL Value</h3>
        <h1 style="margin:5px 0 0 0;">${call_price:.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

with top_col2:
    # Put Value
    st.markdown(
        f"""
        <div style="
            background-color:#f8d7da;
            border:2px solid #dc3545;
            border-radius:15px;
            padding:20px;
            text-align:center;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        ">
        <h3 style="margin:0;">PUT Value</h3>
        <h1 style="margin:5px 0 0 0;">${put_price:.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------
# Bottom row: Separate payoff + "today value" charts
# -------------------------
bottom_col1, bottom_col2 = st.columns(2)

# Make a range of underlying prices around the strike
S_range = np.linspace(0.5 * K, 1.5 * K, 100)

with bottom_col1:
    st.subheader("Call Option: Payoff vs. Value Today")

    # 1) Payoff at expiration
    call_payoff = np.maximum(S_range - K, 0)
    # 2) Black-Scholes value *today* for each S in S_range
    call_value_today = []
    for s_ in S_range:
        c_, _, _, _ = black_scholes(s_, K, T, r, sigma)
        call_value_today.append(c_)

    # Plot them together
    fig_call = go.Figure()
    fig_call.add_trace(go.Scatter(
        x=S_range, y=call_payoff,
        mode='lines', name='Payoff at Expiration',
        line=dict(color='green')
    ))
    fig_call.add_trace(go.Scatter(
        x=S_range, y=call_value_today,
        mode='lines', name='Option Value (Today)',
        line=dict(color='blue')
    ))
    fig_call.update_layout(
        xaxis_title="Underlying Price",
        yaxis_title="Value / Payoff",
        legend=dict(x=0, y=1)
    )
    st.plotly_chart(fig_call, use_container_width=True)



with bottom_col2:
    st.subheader("Put Option: Payoff vs. Value Today")

    # 1) Payoff at expiration
    put_payoff = np.maximum(K - S_range, 0)
    # 2) Black-Scholes value *today* for each S in S_range
    put_value_today = []
    for s_ in S_range:
        _, p_, _, _ = black_scholes(s_, K, T, r, sigma)
        put_value_today.append(p_)

    # Plot them together
    fig_put = go.Figure()
    fig_put.add_trace(go.Scatter(
        x=S_range, y=put_payoff,
        mode='lines', name='Payoff at Expiration',
        line=dict(color='red')
    ))
    fig_put.add_trace(go.Scatter(
        x=S_range, y=put_value_today,
        mode='lines', name='Option Value (Today)',
        line=dict(color='blue')
    ))
    fig_put.update_layout(
        xaxis_title="Underlying Price",
        yaxis_title="Value / Payoff",
        legend=dict(x=0, y=1)
    )
    st.plotly_chart(fig_put, use_container_width=True)



# -------------------------
# Greeks Table
# -------------------------
st.subheader("Option Greeks")
greeks_df = pd.DataFrame({
    "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
    "Call": [
        greeks_dict["Call Delta"],
        greeks_dict["Gamma"],
        greeks_dict["Vega"],
        greeks_dict["Call Theta"],
        greeks_dict["Call Rho"]
    ],
    "Put": [
        greeks_dict["Put Delta"],
        greeks_dict["Gamma"],
        greeks_dict["Vega"],
        greeks_dict["Put Theta"],
        greeks_dict["Put Rho"]
    ]
})
st.table(greeks_df)

# -------------------------
# Expandable Black-Scholes details
# -------------------------
with st.expander("See Black-Scholes Formula Details"):
    st.latex(r"""
    d_1 = \frac{\ln(S / K) + (r + 0.5 \sigma^2) T}{\sigma \sqrt{T}}
    """)
    st.latex(r"""
    d_2 = d_1 - \sigma \sqrt{T}
    """)
    st.latex(r"""
    \text{Call Price} = S\,N(d_1) - K\,e^{-rT}\,N(d_2)
    """)
    st.latex(r"""
    \text{Put Price} = K\,e^{-rT}\,N(-d_2) - S\,N(-d_1)
    """)


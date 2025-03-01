
import streamlit as st 
import numpy as np
from scipy.stats import norm

# Function implementing the Black-Scholes formula
def black_scholes(S, K, T, r, sigma):
    """
    Black-Scholes formula for European Call and Put options.
    S: Spot price
    K: Strike price
    T: Time to maturity
    r: Risk-free interest rate
    sigma: Volatility (standard deviation of returns)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return call_price, put_price

# Streamlit page layout
st.set_page_config(page_title="Black-Scholes Option Pricing U")

st.title("Black-Scholes Option Pricing")

# Sidebar inputs for model parameters
st.sidebar.header("Input Parameters")

S = st.sidebar.number_input("Spot Price (S)", min_value=0.0, value=50.0)
K = st.sidebar.number_input("Strike Price (K)", min_value=0.0, value=50.0)
T = st.sidebar.number_input("Time to Maturity (T, in years)", min_value=0.01, value=1.0, step=0.01)
r = st.sidebar.number_input("Risk-Free Interest Rate (r)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.3, step=0.01)

# Calculate option prices
call_price, put_price = black_scholes(S, K, T, r, sigma)

# CSS styles for the rounded rectangles
rounded_box_css = """
<style>
.box-container {
    display: flex;
    justify-content: center;
    gap: 40px; /* Adjusted gap for better spacing */
    margin-top: 30px;
    flex-wrap: wrap;
}

.rounded-box {
    background-color: #f9f9f9;
    padding: 25px; /* Adjusted padding */
    border-radius: 15px; /* Rounded corners */
    border: 2px solid #ddd;
    width:550px; /* Adjusted width */
    text-align: center;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
}

.call-box {
    background-color: #d4edda;  /* Light green */
    border-color: #28a745;      /* Green border */
}

.put-box {
    background-color: #f8d7da;  /* Light red */
    border-color: #dc3545;      /* Red border */
}

.rounded-box h3 {
    margin: 0;
    font-weight: bold;
    font-size: 25px; /* Increased font size */
    color: #333;
}

.rounded-box h1 {
    margin: 5px 0 0 0;
    font-size: 28px; /* Increased font size */
    color: #333;
}
</style>
"""

# Inject CSS styles
st.markdown(rounded_box_css, unsafe_allow_html=True)

# Display results with custom styled boxes
st.markdown(
    f"""
    <div class="box-container">
        <div class="rounded-box call-box">
            <h3>CALL Value</h3>
            <h1>${call_price:.2f}</h1>
        </div>
        <div class="rounded-box put-box">
            <h3>PUT Value</h3>
            <h1>${put_price:.2f}</h1>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Space between option prices and formula details
st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)

# Optional: Display formula details for context
with st.expander("See Black-Scholes Formula Details"):
    st.latex(r"""
    d_1 = \frac{\ln(S / K) + (r + 0.5 \sigma^2) T}{\sigma \sqrt{T}}
    """)
    st.latex(r"""
    d_2 = d_1 - \sigma \sqrt{T}
    """)
    st.latex(r"""
    \text{Call Price} = S N(d_1) - K e^{-r T} N(d_2)
    """)
    st.latex(r"""
    \text{Put Price} = K e^{-r T} N(-d_2) - S N(-d_1)
    """)


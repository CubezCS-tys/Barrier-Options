# import streamlit as st
# import numpy as np
# from scipy.stats import norm

# # Define helper functions for option pricing

# def calc_d1(S0, K, r, q, sigma, T):
#     d1 = (np.log(S0 / K) + (r - q + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
#     return d1

# def calc_d2(S0, K, r, q, sigma, T):
#     d2 = calc_d1(S0, K, r, q, sigma, T) - sigma * np.sqrt(T)
#     return d2

# def calc_c(S0, K, r, q, sigma, T):
#     d1 = calc_d1(S0, K, r, q, sigma, T)
#     d2 = calc_d2(S0, K, r, q, sigma, T)
#     c = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
#     return c

# def calc_p(S0, K, r, q, sigma, T):
#     d1 = calc_d1(S0, K, r, q, sigma, T)
#     d2 = calc_d2(S0, K, r, q, sigma, T)
#     p = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
#     return p

# def calc_lambda(r, q, sigma):
#     lambda_ = (r - q + (0.5 * (sigma ** 2))) / sigma ** 2
#     return lambda_

# def calc_y(H, S0, K, T, sigma, r, q):
#     lambda_ = calc_lambda(r, q, sigma)
#     y = (np.log((H ** 2) / (S0 * K)) / (sigma * np.sqrt(T))) + lambda_ * sigma * np.sqrt(T)
#     return y

# def calc_x1(S0, H, T, sigma):
#     lambda_ = calc_lambda(r, q, sigma)
#     x1 = ((np.log(S0/H))/(sigma * np.sqrt(T)))+(lambda_ * sigma * np.sqrt(T))
#     return x1

# def calc_y1(S0, H, T, sigma):
#     lambda_ = calc_lambda(r, q, sigma)
#     y1 = ((np.log(H/S0))/(sigma * np.sqrt(T)))+(lambda_ * sigma * np.sqrt(T))
#     return y1

# def barrier_option_price(S0, K, T, r, q, sigma, H, option_type):
#     x1 = calc_x1(S0, H, T, sigma)
#     y1 = calc_y1(S0, H, T, sigma)
#     c = calc_c(S0, K, r, q, sigma, T)
#     lambda_ = calc_lambda(r, q, sigma)
#     y = calc_y(H, S0, K, T, sigma, r, q)
#     p = calc_p(S0, K, r, q, sigma, T)

#     if option_type == 'down-and-in call' and H <= K:
#         cdi = (S0 * np.exp(-q * T) * (H / S0) ** (2 * lambda_) * norm.cdf(y) -
#                K * np.exp(-r * T) * (H / S0) ** (2 * lambda_ - 2) * norm.cdf(y - sigma * np.sqrt(T)))
#         return cdi
#     elif option_type == 'down-and-out call' and H <= K:
#         cdi = (S0 * np.exp(-q * T) * (H / S0) ** (2 * lambda_) * norm.cdf(y) -
#                K * np.exp(-r * T) * (H / S0) ** (2 * lambda_ - 2) * norm.cdf(y - sigma * np.sqrt(T)))
#         cdo = c - cdi
#         return cdo
#     elif option_type == 'down-and-out call' and H >= K:
#         cdo = (S0 * norm.cdf(x1) * np.exp(-q * T)) - (K * np.exp(-r * T ) * norm.cdf(x1 - (sigma * np.sqrt(T)))) - (S0 *np.exp(-q * T)*((H/S0)**(2*lambda_)) * norm.cdf(y1)) + (K * np.exp(-r * T) * ((H/S0)**((2*lambda_) - 2)) * norm.cdf(y1 - (sigma * np.sqrt(T))))
#         # term1 = S0 * np.exp(-q * T) * norm.cdf(x1)
#         # term2 = K * np.exp(-r * T) * norm.cdf(x1 - sigma * np.sqrt(T))
#         # term3 = S0 * np.exp(-q * T) * (H / S0)**(2 * lambda_) * norm.cdf(y1)
#         # term4 = K * np.exp(-r * T) * (H / S0)**(2 * lambda_ - 2) * norm.cdf(y1 - sigma * np.sqrt(T))
    
#         # Calculate the option price
#         #cdo = term1 - term2 - term3 + term4
#         if cdo < 0:
#             return 0
#         else:
#             return cdo

#     elif option_type == 'down-and-in call' and H >= K:
#          cdo = (S0 * norm.cdf(x1) * np.exp(-q * T)) - (K * np.exp(-r * T ) * norm.cdf(x1 - (sigma * np.sqrt(T)))) - (S0 *np.exp(-q * T)*((H/S0)**2*lambda_) * norm.cdf(y1)) + (K * np.exp(-r * T) * ((H/S0)**((2*lambda_) - 2)) * norm.cdf(y1 - (sigma * np.sqrt(T))))
#          cdi = c - cdo
#          if cdo < 0:
#             cdo = 0
#             cdi   = c - cdo
#             return cdi
#          else:
#             return cdi
#     elif option_type == 'up-and-in call' and H > K:
#         cui = (S0 * norm.cdf(x1) * np.exp(-q * T)) - (K * np.exp(-r * T ) * norm.cdf(x1 - (sigma * np.sqrt(T)))) - (S0 * np.exp(-q * T) * ((H/S0)**2*lambda_) * (norm.cdf(-y) - norm.cdf(-y1))) + (K * np.exp(-r * T) * ((H/S0)**(2*lambda_-2)) * (norm.cdf(-y + (sigma * np.sqrt(T))) - norm.cdf(-y1 + (sigma * np.sqrt(T)))))
#         return cui
#     elif option_type == 'up-and-in call' and H <= K:
#         cui = c
#         return cui
#     elif option_type == 'up-and-out call' and H <= K:
#         cuo = 0
#         return cuo
#     elif option_type == 'up-and-out call' and H > K:
#         cui = (S0 * norm.cdf(x1) * np.exp(-q * T)) - (K * np.exp(-r * T ) * norm.cdf(x1 - (sigma * np.sqrt(T)))) - (S0 * np.exp(-q * T) * ((H/S0)**2*lambda_) * (norm.cdf(-y) - norm.cdf(-y1))) + (K * np.exp(-r * T) * ((H/S0)**(2*lambda_-2)) * (norm.cdf(-y + (sigma * np.sqrt(T))) - norm.cdf(-y1 + (sigma * np.sqrt(T)))))
#         cuo = c - cui
#         return cuo
#         #if 
#     elif option_type == 'up-and-in put' and H >= K:
#         pui = (-S0 * np.exp(-q * T) * ((H/S0)**2*lambda_) * norm.cdf(-y)) + (K * np.exp(-r * T) * ((H/S0)**((2*lambda_)-2)) * norm.cdf(-y + (sigma * np.sqrt(T))))
#         return pui
#     elif option_type == 'up-and-out put' and H >= K:
#         pui = (-S0 * np.exp(-q * T) * ((H/S0)**2*lambda_) * norm.cdf(-y)) + (K * np.exp(-r * T) * ((H/S0)**((2*lambda_)-2)) * norm.cdf(-y + (sigma * np.sqrt(T))))
#         puo = p - pui
#         return puo
#     elif option_type == 'up-and-out put' and H <= K: ###
#         # puo = (-S0 * norm.cdf(-x1) * np.exp(-q * T)) + (K * np.exp(-r * T) * norm.cdf(-x1 + (sigma * np.sqrt(T)))) + (S0 * np.exp(-q * T) * ((H/S0)**2*lambda_) * norm.cdf(-y1)) - (K * np.exp(-r * T) * ((H/S0)**(2*lambda_ - 2)) * norm.cdf(-y1 + (sigma* np.sqrt(T))))
#         # puo = (-S0 * norm.cdf(-x1) * np.exp(-q * T)) + (K * np.exp(-r * T ) * norm.cdf(-x1 + (sigma * np.sqrt(T)))) + (S0 *np.exp(-q * T)*((H/S0)**2*lambda_) * norm.cdf(-y1)) - (K * np.exp(-r * T) * ((H/S0)**((2*lambda_) - 2)) * norm.cdf(-y1 + (sigma * np.sqrt(T))))
#         puo = (-S0 * norm.cdf(-x1) * np.exp(-q * T)) + (K * np.exp(-r * T ) * norm.cdf(-x1 + (sigma * np.sqrt(T)))) + (S0 *np.exp(-q * T)*((H/S0)**2*lambda_) *(norm.cdf(-y1))) -(K * np.exp(-r * T) * ((H/S0)**(2*lambda_-2)) * (norm.cdf(-y1 + (sigma * np.sqrt(T)))))
#         if puo < 0:
#             return 0
#         else:
#             return puo
#     elif option_type == 'up-and-in put' and H <= K:
#         # puo = (-S0 * norm.cdf(-x1) * np.exp(-q * T)) + (K * np.exp(-r * T) * norm.cdf(-x1 + (sigma * np.sqrt(T)))) + (S0 * np.exp(-q * T) * ((H/S0)**2*lambda_) * norm.cdf(-y1)) - (K * np.exp(-r * T) * ((H/S0)**(2*lambda_ - 2)) * norm.cdf(-y1 + (sigma* np.sqrt(T)))) 
#         #puo = (-S0 * norm.cdf(-x1) * np.exp(-q * T)) + (K * np.exp(-r * T ) * norm.cdf(-x1 + (sigma * np.sqrt(T)))) + (S0 *np.exp(-q * T)*((H/S0)**2*lambda_) * norm.cdf(-y1)) - (K * np.exp(-r * T) * ((H/S0)**((2*lambda_) - 2)) * norm.cdf(-y1 + (sigma * np.sqrt(T))))
#         puo = (-S0 * norm.cdf(-x1) * np.exp(-q * T)) + (K * np.exp(-r * T ) * norm.cdf(-x1 + (sigma * np.sqrt(T)))) + (S0 *np.exp(-q * T)*((H/S0)**2*lambda_) *(norm.cdf(-y1))) -(K * np.exp(-r * T) * ((H/S0)**(2*lambda_-2)) * (norm.cdf(-y1 + (sigma * np.sqrt(T)))))
#         if puo < 0:
#             puo = 0
#             pui = p - puo
#             return pui
#         else:
#             pui = p -puo
#             return pui 
#     elif option_type == 'down-and-out put' and H > K:
#         pdo = 0
#         return pdo
#     elif option_type == 'down-and-in put' and H > K:
#         pdi = p
#         return pdi
#     elif option_type == 'down-and-in put' and H < K:
#         pdi = (-S0 * norm.cdf(-x1) * np.exp(-q * T)) + (K * np.exp(-r * T) * norm.cdf(-x1 + (sigma * np.sqrt(T)))) + (S0 * np.exp(-q * T) * ((H/S0)**2*lambda_) * (norm.cdf(y) - norm.cdf(y1))) - (K * np.exp(-r * T) * ((H/S0)**(2*lambda_-2)) * (norm.cdf(y - (sigma * np.sqrt(T))) - norm.cdf(y1 - (sigma * np.sqrt(T)))))
#         return pdi
#     elif option_type == 'down-and-out put' and H < K:
#         pdi = (-S0 * norm.cdf(-x1) * np.exp(-q * T)) + (K * np.exp(-r * T) * norm.cdf(-x1 + (sigma * np.sqrt(T)))) + (S0 * np.exp(-q * T) * ((H/S0)**2*lambda_) * (norm.cdf(y) - norm.cdf(y1))) - (K * np.exp(-r * T) * ((H/S0)**(2*lambda_-2)) * (norm.cdf(y - (sigma * np.sqrt(T))) - norm.cdf(y1 - (sigma * np.sqrt(T)))))
#         pdo = p - pdi
#         return pdo
#     else:
#         return None

# # Streamlit UI
# st.title("Barrier Option Pricing")
# st.sidebar.header("Input Parameters")

# S0 = st.sidebar.number_input("Stock Price (S0)", value=100.0)
# K = st.sidebar.number_input("Strike Price (K)", value=100.0)
# T = st.sidebar.number_input("Term (Years)", value=1.0)
# r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05)
# q = st.sidebar.number_input("Dividend Yield (q)", value=0.03)
# sigma = st.sidebar.number_input("Volatility (σ)", value=0.2)
# H = st.sidebar.number_input("Barrier (H)", value=80.0)
# rebate = st.sidebar.number_input("Rebate", value=0.0)

# option_type = st.sidebar.selectbox("Option Type", ["down-and-in call", "down-and-out call", "down-and-out put", "up-and-in call", "up-and-out call", "up-and-in put", "up-and-out put", "down-and-in put"])

# # Calculate and display option price
# option_price = barrier_option_price(S0, K, T, r, q, sigma, H, option_type)
# if option_price is not None:
#     st.write(f"**{option_type.capitalize()} Price:**${option_price:.2f}")
# else:
#     st.write("Invalid option type selected or option type not implemented.")
#####################################################################################################################################################################################
import streamlit as st
import numpy as np
from scipy.stats import norm

import streamlit as st

st.title("Barrier Option Pricing Equations")

st.write("""
### Regular Call and Put Options
The prices at time zero of a regular European call option ($c$) and put option ($p$) are given by:
""")
st.latex(r"""
c = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)
""")
st.latex(r"""
p = K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)
""")
st.write("Where:")
st.latex(r"""
d_1 = \frac{\ln(S_0/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad
d_2 = d_1 - \sigma\sqrt{T}
""")

st.write("""
### Down-and-In Call Option for $H$ $\leq$ $K$
""")
st.latex(r"""
c_{di} = S_0 e^{-qT} \left(\frac{H}{S_0}\right)^{2\lambda} N(y) - K e^{-rT} \left(\frac{H}{S_0}\right)^{2\lambda-2} N(y - \sigma\sqrt{T})
""")
st.latex(r"""
\text{Where: } 
\lambda = \frac{r - q + \sigma^2/2}{\sigma^2}, \quad
y = \frac{\ln(H^2 / (S_0 K))}{\sigma\sqrt{T}} + \lambda\sigma\sqrt{T}
""")

st.write("""
### Down-and-Out Call Option for $H$ $\leq$ $K$
""")
st.latex(r"""
c_{do} = c - c_{di}
""")

st.write("""
### Down-and-Out Call Option for $H$ $\geq$ $K$
""")

st.latex(r"""
c_{do} = S_0 N(x_1) e^{-qT} - K e^{-rT} N(x_1 - \sigma\sqrt{T}) 
        - S_0 e^{-qT} \left(\frac{H}{S_0}\right)^{2\lambda} N(y_1) 
        + K e^{-rT} \left(\frac{H}{S_0}\right)^{2\lambda-2} N(y_1 - \sigma\sqrt{T})
""")
st.latex(r"""
\text{Where: }
x_1 = \frac{\ln(S_0 / H)}{\sigma\sqrt{T}} + \lambda\sigma\sqrt{T}, \quad
y_1 = \frac{\ln(H / S_0)}{\sigma\sqrt{T}} + \lambda\sigma\sqrt{T}
""")

st.write("""
### Down-and-In Call Option for $H$ $\geq$ $K$
""")

st.latex(r"""
c_{di} = c - c_{do}
""")


st.write("""
### Up-and-In Call Option for $H$ $\geq$ $K$

""")
st.latex(r"""
c_{ui} = S_0 N(x_1) e^{-qT} - K e^{-rT} N(x_1 - \sigma\sqrt{T}) 
        - S_0 e^{-qT} \left(\frac{H}{S_0}\right)^{2\lambda}[N(-y) - N(-y_1)] + K e^{-rT} \left(\frac{H}{S_0}\right)^{2\lambda-2} [N(-y + \sigma\sqrt{T} - N(-y_1 + \sigma\sqrt{T})]
""")

st.write("""
### Up-and-Out Call Option for $H$ $\geq$ $K$
""")
st.latex(r"""
c_{uo} = c - c_{ui}
""")

st.write(""" 
         ### Up-and-In Put Option for $H$ $\geq$ $K$
         """)
st.latex(r"""
p_{ui} = -S_0 e^{-qT} \left(\frac{H}{S_0}\right)^{2\lambda} N(-y) 
         + K e^{-rT} \left(\frac{H}{S_0}\right)^{2\lambda-2} N(-y + \sigma\sqrt{T})
""")

st.write(""" 
         ### Up-and-Out Put Option for $H$ $\geq$ $K$
         """)

st.latex(r"""
p_{uo} = p - p_{ui}
""")

st.write(""" 
         ### Up-and-Out Put Option for $H$ $\leq$ $K$
         """)
st.latex(r"""
    p_{uo} = -S_0 N(-x_1) e^{-qT} + K e^{-rT} N(-x_1 + \sigma\sqrt{T}) 
         + S_0 e^{-qT} \left(\frac{H}{S_0}\right)^{2\lambda} N(-y_1) 
         - K e^{-rT} \left(\frac{H}{S_0}\right)^{2\lambda-2} N(-y_1 + \sigma\sqrt{T})
         """)

st.write(""" 
         ### Up-and-In Put Option for $H$ $\leq$ $K$
         """)
st.latex(r"""
p_{ui} = p - p_{uo}
""")

st.write(""" 
         ### Down-and-Out Put Option for $H$ $\geq$ $K$
         """)
st.latex(r"""
p_{do} = 0
""")

st.write(""" 
         ### Down-and-In Put Option for $H$ $\geq$ $K$
         """)
st.latex(r"""
p_{di} = p
""")

st.write(""" 
         ### Down-and-In Put Option for $H$ $\leq$ $K$
         """)

st.latex(r"""
    p_{uo} = -S_0 N(-x_1) e^{-qT} + K e^{-rT} N(-x_1 + \sigma\sqrt{T}) 
         + S_0 e^{-qT} \left(\frac{H}{S_0}\right)^{2\lambda} [N(y) - N(y_1)] - K e^{-rT} \left(\frac{H}{S_0}\right)^{2\lambda - 2} [N(y - \sigma\sqrt{T}) - N(y_1 - \sigma\sqrt{T})]
         """)

st.write(""" 
         ### Down-and-Out Put Option for $H$ $\leq$ $K$
         """)
st.latex(r"""
p_{do} = p - p_{di}
""")






# ------------------------------
# 1) Helper functions
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
    # λ = (r - q + σ²/2) / σ²
    return (r - q + 0.5 * sigma**2) / (sigma**2)

def calc_y(H, S0, K, T, sigma, r, q):
    """
    y = [ln(H^2/(S0*K)) / (sigma*sqrt(T))] + λ * sigma * sqrt(T)
    """
    lam = calc_lambda(r, q, sigma)
    return (np.log((H**2)/(S0*K)) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

def calc_x1(S0, H, T, sigma, r, q):
    """
    x1 = ln(S0/H)/(sigma*sqrt(T)) + λ*sigma*sqrt(T)
    """
    lam = calc_lambda(r, q, sigma)
    return (np.log(S0/H) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

def calc_y1(S0, H, T, sigma, r, q):
    """
    y1 = ln(H/S0)/(sigma*sqrt(T)) + λ*sigma*sqrt(T)
    """
    lam = calc_lambda(r, q, sigma)
    return (np.log(H/S0) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

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

def barrier_option_price(S0, K, T, r, q, sigma, H, option_type):
    """
    Returns the price of a barrier option (various knock-in/out types).
    Matches standard formulas from texts like Hull, with care to keep
    exponents and sign conventions correct.
    """
    x1 = calc_x1(S0, H, T, sigma, r, q)
    y1 = calc_y1(S0, H, T, sigma, r, q)
    c = calc_c(S0, K, r, q, sigma, T)
    p = calc_p(S0, K, r, q, sigma, T)
    lam = calc_lambda(r, q, sigma)
    y  = calc_y(H, S0, K, T, sigma, r, q)

    # --------------------------------
    # Down-and-in Call
    # --------------------------------
    
    if option_type == 'down-and-in call' and H <= K and S0 <= H:
        vanilla = black_scholes(S0, K, T, r, sigma, "Call")
        return vanilla
    
    elif option_type == 'down-and-in call' and H <= K:
        # cdi, for H <= K
        cdi = (S0 * np.exp(-q*T) * (H/S0)**(2*lam) * norm.cdf(y)
               - K * np.exp(-r*T) * (H/S0)**(2*lam - 2)
                 * norm.cdf(y - sigma*np.sqrt(T)))
        return cdi

    elif option_type == 'down-and-in call' and H >= K:
        # cdi = c - cdo. So we compute cdo from the standard expression
        # cdo = ...
        # Then cdi = c - cdo
        term1 = S0*np.exp(-q*T)*norm.cdf(x1)
        term2 = K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
        term3 = S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(y1)
        term4 = K*np.exp(-r*T)*(H/S0)**(2*lam - 2)*norm.cdf(y1 - sigma*np.sqrt(T))
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
    elif option_type == 'down-and-out call' and H <= K:

        cdi = (S0 * np.exp(-q*T) * (H/S0)**(2*lam) * norm.cdf(y)
            - K * np.exp(-r*T) * (H/S0)**(2*lam - 2)
                * norm.cdf(y - sigma*np.sqrt(T)))
        cdo = c - cdi
        if cdo > 0:
            return cdo
        else:
            return 0

    elif option_type == 'down-and-out call' and H >= K:
        # This is the “If H > K” formula for the down-and-out call
        term1 = S0 * np.exp(-q*T)*norm.cdf(x1)
        term2 = K  * np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
        term3 = S0 * np.exp(-q*T)*((H/S0)**(2*lam))*norm.cdf(y1)
        term4 = K  * np.exp(-r*T)*((H/S0)**(2*lam - 2))*norm.cdf(y1 - sigma*np.sqrt(T))
        cdo   = term1 - term2 - term3 + term4
        
        if cdo < 0:
            return 0
        else:
            return cdo

    # --------------------------------
    # Up-and-in Call
    # --------------------------------
    elif option_type == 'up-and-in call' and H > K:
        # Standard up-and-in call for H > K
        cui = (S0*np.exp(-q*T)*norm.cdf(x1)
               - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
               - S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(-y) - norm.cdf(-y1))
               + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
                 * (norm.cdf(-y + sigma*np.sqrt(T))
                    - norm.cdf(-y1 + sigma*np.sqrt(T))))
        return cui

    elif option_type == 'up-and-in call' and H <= K:
        # If barrier is below K, the up-and-in call is effectively the same as c
        # or 0, depending on your setup.  Typically if H < S0 < K,
        # the option knocks in only if S0 goes above H.  If you are sure
        # you want to treat it as simply c, do so here:
        return c

    # --------------------------------
    # Up-and-out Call
    # --------------------------------
    elif option_type == 'up-and-out call' and H <= K:
        # If the barrier H <= K is below the current spot,
        # often up-and-out call is worthless if it is truly "up" barrier?
        return 0.0

    elif option_type == 'up-and-out call' and H > K:
        cui = (S0*np.exp(-q*T)*norm.cdf(x1)
               - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
               - S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(-y) - norm.cdf(-y1))
               + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
                 * (norm.cdf(-y + sigma*np.sqrt(T))
                    - norm.cdf(-y1 + sigma*np.sqrt(T))))
        cuo = c - cui
        return cuo

    # --------------------------------
    # Up-and-in Put
    # --------------------------------
    elif option_type == 'up-and-in put' and H >= K:
        pui = (-S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y)
               + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
                 * norm.cdf(-y + sigma*np.sqrt(T)))
        return pui
    
        # --------------------------------
    elif option_type == 'up-and-in put' and H <= K:
        puo = (
            -S0*np.exp(-q*T)*norm.cdf(-x1)
            + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
            + S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y1)
            - K*np.exp(-r*T)*(H/S0)**(2*lam - 2)*norm.cdf(-y1 + sigma*np.sqrt(T))
        )
        if puo < 0:
            puo = 0
            pui = black_scholes(S0,K,T,r,sigma,"Put")
            return pui
        else:
            pui = black_scholes(S0,K,T,r,sigma,"Put") - puo
        
        return pui
    
    elif option_type == 'up-and-in put' and H <= K:
        # up-and-in put is the difference p - up-and-out put
        # but for the simplified logic, we can just return p if the barrier is < K
        return p

    # --------------------------------
    # Up-and-out Put
    # --------------------------------
    elif option_type == 'up-and-out put' and H >= K:
        # puo = p - pui
        pui = (-S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y)
               + K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
                 * norm.cdf(-y + sigma*np.sqrt(T)))
        puo = p - pui
        return puo

    elif option_type == 'up-and-out put' and H <= K:
        # Standard formula for H <= K
        puo = (
            -S0*np.exp(-q*T)*norm.cdf(-x1)
            + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
            + S0*np.exp(-q*T)*(H/S0)**(2*lam)*norm.cdf(-y1)
            - K*np.exp(-r*T)*(H/S0)**(2*lam - 2)*norm.cdf(-y1 + sigma*np.sqrt(T))
        )
        if puo < 0:
            return 0
        else:
            return puo

    # --------------------------------
    # Down-and-in Put
    # --------------------------------
    elif option_type == 'down-and-in put' and H < K and S0 < H:
        vanilla = black_scholes(S0, K, T, r, sigma, "Put")
        return vanilla
    
    elif option_type == 'down-and-in put' and H > K:
        # If the barrier is above K, we often treat the down-and-in put as p
        return p

    elif option_type == 'down-and-in put' and H < K:
        pdi = (
            -S0*np.exp(-q*T)*norm.cdf(-x1)
            + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
            + S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(y) - norm.cdf(y1))
            - K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
              * (norm.cdf(y - sigma*np.sqrt(T))
                 - norm.cdf(y1 - sigma*np.sqrt(T)))
        )
        return pdi

    # --------------------------------
    # Down-and-out Put
    # --------------------------------
    elif option_type == 'down-and-out put' and H > K:
        # Typically worthless if H > K in certain setups
        return 0

    elif option_type == 'down-and-out put' and H < K:
        pdi = (
            -S0*np.exp(-q*T)*norm.cdf(-x1)
            + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
            + S0*np.exp(-q*T)*(H/S0)**(2*lam)*(norm.cdf(y) - norm.cdf(y1))
            - K*np.exp(-r*T)*(H/S0)**(2*lam - 2)
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
# 3) Streamlit UI
# ------------------------------

st.title("Barrier Option Pricing")
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

option_price = barrier_option_price(S0, K, T, r, q, sigma, H, option_type)
if option_price is not None:
    st.write(f"**{option_type.capitalize()} Price:** ${option_price:.4f}")
else:
    st.write("Invalid option type selected or option type not implemented.")

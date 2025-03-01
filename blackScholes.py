
# Simple black scholes implementation
import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma):
    """
    Black-Scholes formula for European Call and Put options
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

# Example Usage
S = 50  # Spot price
K = 50  # Strike price
T = 1    # Time to maturity in years
r = 0.05 # Risk-free rate
sigma = 0.3  # Volatility

call_price, put_price = black_scholes(S, K, T, r, sigma)
print(f"Call Price: {call_price:.4f}")
print(f"Put Price: {put_price:.4f}")


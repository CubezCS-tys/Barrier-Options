import numpy as np
from scipy.stats import norm

def barrier_exact(s, K, B, r, sigma, T):
    """
    Calculates the exact price of a down-and-out European call option.

    Parameters:
    - s: Current stock price
    - K: Strike price
    - B: Barrier level
    - r: Risk-free interest rate
    - sigma: Volatility of the underlying asset
    - T: Time to maturity (in years)

    Returns:
    - exactValue: The exact price of the barrier option
    """
    # Calculate auxiliary variables
    a = (B / s) ** (-1 + (2 * r) / sigma ** 2)
    b = (B / s) ** (1 + (2 * r) / sigma ** 2)

    d1 = (np.log(s / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(s / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d3 = (np.log(s / B) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d4 = (np.log(s / B) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d5 = (np.log(s / B) - (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d6 = (np.log(s / B) - (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d7 = (np.log(s / B ** 2) - (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d8 = (np.log(s / B ** 2) - (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    # Calculate the exact value
    exactValue = s * (norm.cdf(d1) - norm.cdf(d3) - b * (norm.cdf(d6) - norm.cdf(d8))) \
                 - K * np.exp(-r * T) * (norm.cdf(d2) - norm.cdf(d4) - a * (norm.cdf(d5) - norm.cdf(d7)))

    return exactValue

print(barrier_exact(100,100,120,0.05,0.2,1))
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Black-Scholes formula for European call and put options
def black_scholes(S, K, T, r, sigma):
    """
    Black-Scholes formula for European Call and Put options.
    Parameters:
    S (float): Spot price
    K (float): Strike price
    T (float): Time to maturity (in years)
    r (float): Risk-free interest rate (as a decimal)
    sigma (float): Volatility (standard deviation of returns, as a decimal)
    
    Returns:
    tuple: (call_price, put_price)
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0, 0.0  # Handle invalid inputs gracefully
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return call_price, put_price

# Initial Parameters
initial_S = 50       # Spot price
initial_K = 50       # Strike price
initial_T = 1        # Time to maturity (in years)
initial_r = 0.05     # Risk-free interest rate (as a decimal)
initial_sigma = 0.3  # Volatility (as a decimal)

# --- Graph 1: Option Price vs Spot Price ---
spot_prices = np.linspace(0.01, 100, 100)
call_prices = []
put_prices = []

for S_value in spot_prices:
    call_price, put_price = black_scholes(S_value, initial_K, initial_T, initial_r, initial_sigma)
    call_prices.append(call_price)
    put_prices.append(put_price)

plt.figure(figsize=(10, 6))
plt.plot(spot_prices, call_prices, label='Call Option Price', color='blue')
plt.plot(spot_prices, put_prices, label='Put Option Price', color='red')
plt.title('Option Price vs Spot Price')
plt.xlabel('Spot Price (S)')
plt.ylabel('Option Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Graph 2: Option Price vs Time to Maturity ---
time_to_maturity = np.linspace(0.01, 1.8, 100)
call_prices_time = []
put_prices_time = []

for T_value in time_to_maturity:
    call_price, put_price = black_scholes(initial_S, initial_K, T_value, initial_r, initial_sigma)
    call_prices_time.append(call_price)
    put_prices_time.append(put_price)

plt.figure(figsize=(10, 6))
plt.plot(time_to_maturity, call_prices_time, label='Call Option Price', color='blue')
plt.plot(time_to_maturity, put_prices_time, label='Put Option Price', color='red')
plt.title('Option Price vs Time to Maturity')
plt.xlabel('Time to Maturity (T)')
plt.ylabel('Option Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Graph 3: Option Price vs Strike Price ---
strike_prices = np.linspace(0.01, 100, 100)
call_prices_strike = []
put_prices_strike = []

for K_value in strike_prices:
    call_price, put_price = black_scholes(initial_S, K_value, initial_T, initial_r, initial_sigma)
    call_prices_strike.append(call_price)
    put_prices_strike.append(put_price)

plt.figure(figsize=(10, 6))
plt.plot(strike_prices, call_prices_strike, label='Call Option Price', color='blue')
plt.plot(strike_prices, put_prices_strike, label='Put Option Price', color='red')
plt.title('Option Price vs Strike Price')
plt.xlabel('Strike Price (K)')
plt.ylabel('Option Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Graph 4: Option Price vs Volatility ---
volatilities = np.linspace(0.01, 1.0, 100)  # Increased to 100 points for smoother curve
call_prices_volatility = []
put_prices_volatility = []

for sigma_value in volatilities:
    call_price, put_price = black_scholes(initial_S, initial_K, initial_T, initial_r, sigma_value)
    call_prices_volatility.append(call_price)
    put_prices_volatility.append(put_price)

plt.figure(figsize=(10, 6))
plt.plot(volatilities * 100, call_prices_volatility, label='Call Option Price', color='blue')
plt.plot(volatilities * 100, put_prices_volatility, label='Put Option Price', color='red')
plt.title('Option Price vs Volatility')
plt.xlabel('Volatility (%)')
plt.ylabel('Option Price')
plt.legend()
plt.grid(True)
plt.ylim(bottom=0)  # Ensure y-axis starts at zero
plt.tight_layout()
plt.show()

# --- Graph 5: Option Price vs Risk-Free Interest Rate ---
interest_rates = np.linspace(0.001, 0.08, 100)
call_prices_rate = []
put_prices_rate = []

for r_value in interest_rates:
    call_price, put_price = black_scholes(initial_S, initial_K, initial_T, r_value, initial_sigma)
    call_prices_rate.append(call_price)
    put_prices_rate.append(put_price)

plt.figure(figsize=(10, 6))
plt.plot(interest_rates * 100, call_prices_rate, label='Call Option Price', color='blue')
plt.plot(interest_rates * 100, put_prices_rate, label='Put Option Price', color='red')
plt.title('Option Price vs Risk-Free Interest Rate')
plt.xlabel('Risk-Free Interest Rate (%)')
plt.ylabel('Option Price')
plt.legend()
plt.grid(True)
plt.ylim(bottom=0)  # Ensure y-axis starts at zero
plt.tight_layout()
plt.show()



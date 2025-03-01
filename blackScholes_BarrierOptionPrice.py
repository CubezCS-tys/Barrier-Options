import numpy as np

def barrier_option_pricer(S0, K, T, r, sigma, B, option_type="call", barrier_type="up-and-in", num_simulations=10000, num_steps=100):
    dt = T / num_steps
    payoffs = []

    for _ in range(num_simulations):
        # Simulate the path
        S_t = S0
        path = [S_t]
        barrier_triggered = False

        for _ in range(num_steps):
            Z = np.random.standard_normal()
            S_t *= np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
            path.append(S_t)

            # Check barrier conditions
            if barrier_type == "up-and-in" and S_t >= B:
                barrier_triggered = True
            elif barrier_type == "down-and-in" and S_t <= B:
                barrier_triggered = True
            elif barrier_type == "up-and-out" and S_t >= B:
                barrier_triggered = False
                break
            elif barrier_type == "down-and-out" and S_t <= B:
                barrier_triggered = False
                break

        # Determine if option is active
        if ("in" in barrier_type and barrier_triggered) or ("out" in barrier_type and not barrier_triggered):
            # Calculate payoff at maturity
            if option_type == "call":
                payoff = max(S_t - K, 0)
            else:  # put option
                payoff = max(K - S_t, 0)
            payoffs.append(payoff)

    # Discount the average payoff back to the present value
    option_price = np.exp(-r * T) * np.mean(payoffs) if payoffs else 0
    return option_price

# Parameters
S0 = 100       # Initial stock price
K = 100        # Strike price
T = 1.0        # Time to maturity (1 year)
r = 0.05       # Risk-free interest rate
sigma = 0.2    # Volatility
B = 120        # Barrier level
num_simulations = 10000
num_steps = 100

# Pricing different types of barrier options
option_types = ["call", "put"]
barrier_types = ["up-and-in", "down-and-in", "up-and-out", "down-and-out"]

for option_type in option_types:
    for barrier_type in barrier_types:
        price = barrier_option_pricer(S0, K, T, r, sigma, B, option_type, barrier_type, num_simulations, num_steps)
        print(f"The price of a {barrier_type} {option_type} option is: ${price:.2f}")

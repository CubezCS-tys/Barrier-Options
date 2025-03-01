import numpy as np
import plotly.graph_objects as go
import plotly.colors as colors

# Parameters
S0 = 100       # Initial stock price
K = 110        # Strike price
T = 10.0        # Time to maturity (1 year)
r = 0.05       # Risk-free interest rate
sigma = 0.2    # Volatility
num_simulations = 1000         # Number of Monte Carlo simulations
num_steps = 1000              # Number of time steps

# Time delta
dt = T / num_steps

# Simulating price paths
np.random.seed(0)  # For reproducibility
price_paths = np.zeros((num_steps + 1, num_simulations))
price_paths[0] = S0

for t in range(1, num_steps + 1):
    Z = np.random.standard_normal(num_simulations)
    price_paths[t] = price_paths[t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

# Calculate the option price based on the terminal prices
payoffs = np.maximum(price_paths[-1] - K, 0)
call_price = np.exp(-r * T) * np.mean(payoffs)

print(f"The Monte Carlo estimate for the European call option price is: {call_price:.2f}")

# Visualization using Plotly
fig = go.Figure()

# Define a color palette for bolder lines
color_palette = colors.qualitative.Bold

# Plot each price path with a bolder color scheme
for i in range(num_simulations):
    fig.add_trace(go.Scatter(
        x=np.linspace(0, T, num_steps + 1),
        y=price_paths[:, i],
        mode='lines',
        line=dict(color=color_palette[i % len(color_palette)], width=1.5),
        opacity=0.7  # Slightly increase opacity for clarity
    ))

# Customize the plot
fig.update_layout(
    title="Simulated Stock Price Paths Using Monte Carlo",
    xaxis_title="Time (Years)",
    yaxis_title="Stock Price",
    showlegend=False
)

# Show plot
fig.show()

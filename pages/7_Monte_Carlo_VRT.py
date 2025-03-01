# import streamlit as st
# import numpy as np
# import plotly.graph_objects as go
# import plotly.colors as colors
# from scipy.stats import norm

# # Streamlit app title
# st.set_page_config(page_title="Monte Carlo Simulation with Variance Reduction", layout="wide")
# st.title("Monte Carlo Simulation for European Call Option with Variance Reduction")

# # Sidebar inputs for model parameters
# st.sidebar.header("Input Parameters")

# # Input parameters
# S0 = st.sidebar.number_input("Initial Stock Price (S0)", value=100.0, step=1.0)
# K = st.sidebar.number_input("Strike Price (K)", value=110.0, step=1.0)
# T = st.sidebar.number_input("Time to Maturity in Years (T)", value=1.0, step=0.1)
# r = st.sidebar.number_input("Risk-Free Interest Rate (r)", value=0.05, step=0.01)
# sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01)
# num_simulations = st.sidebar.number_input("Number of Simulations", value=10000, step=1000)
# num_steps = st.sidebar.number_input("Number of Steps per Simulation", value=100, step=10)

# # Variance Reduction Techniques
# st.sidebar.header("Variance Reduction Techniques")
# use_antithetic = st.sidebar.checkbox("Use Antithetic Variates")
# use_control_variate = st.sidebar.checkbox("Use Control Variates")

# # Function to calculate Black-Scholes price for a European Call Option
# def black_scholes_call_price(S, K, T, r, sigma):
#     d1 = (np.log(S / K) + (r + 0.5 * sigma **2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
#     call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
#     return call_price

# # Run the Monte Carlo simulation when the user clicks the button
# if st.button("Run Simulation"):
#     dt = T / num_steps
#     # Initialize arrays
#     price_paths = np.zeros((num_steps + 1, num_simulations))
#     price_paths[0] = S0

#     # Antithetic Variates
#     if use_antithetic:
#         num_simulations = num_simulations // 2 * 2  # Ensure even number
#         half_simulations = num_simulations // 2
#         Z = np.random.standard_normal((num_steps, half_simulations))
#         Z_antithetic = -Z
#         Z = np.concatenate((Z, Z_antithetic), axis=1)
#     else:
#         Z = np.random.standard_normal((num_steps, num_simulations))

#     # Simulate price paths
#     for t in range(1, num_steps + 1):
#         if t == 1:
#             prev_prices = np.full(num_simulations, S0)
#         else:
#             prev_prices = price_paths[t - 1]
#         drift = (r - 0.5 * sigma ** 2) * dt
#         diffusion = sigma * np.sqrt(dt) * Z[t - 1]
#         price_paths[t] = prev_prices * np.exp(drift + diffusion)

#     # Calculate the payoff for each path at maturity
#     payoffs = np.exp(-r * T) * np.maximum(price_paths[-1] - K, 0)

#     # Control Variates
#     if use_control_variate:
#         # Calculate the analytical Black-Scholes price
#         bs_price = black_scholes_call_price(S0, K, T, r, sigma)
#         # Calculate control variate (terminal stock price)
#         control_variate = price_paths[-1]
#         # Calculate covariance and variance
#         cov_matrix = np.cov(payoffs, control_variate)
#         cov_xy = cov_matrix[0, 1]
#         var_y = cov_matrix[1, 1]
#         # Optimal coefficient
#         beta = -cov_xy / var_y
#         # Adjust payoffs
#         adjusted_payoffs = payoffs + beta * (control_variate - S0 * np.exp(r * T))
#         # Estimate option price
#         call_price = np.mean(adjusted_payoffs)
#     else:
#         # Estimate option price without control variate
#         call_price = np.mean(payoffs)

#     # Display the calculated option price
#     st.write(f"Monte Carlo Estimated Price for the European Call Option: ${call_price:.4f}")

#     # Calculate standard error
#     std_error = np.std(payoffs) / np.sqrt(num_simulations)
#     st.write(f"Standard Error: ${std_error:.4f}")

#     # Analytical Black-Scholes price
#     bs_price = black_scholes_call_price(S0, K, T, r, sigma)
#     st.write(f"Analytical Black-Scholes Price: ${bs_price:.4f}")

#     # Plot the simulated paths
#     fig = go.Figure()
#     # Define a color palette for bolder lines
#     color_palette = colors.qualitative.Bold
#     for i in range(min(100, num_simulations)):  # Display up to 100 paths for clarity
#         fig.add_trace(go.Scatter(
#             x=np.linspace(0, T, num_steps + 1),
#             y=price_paths[:, i],
#             mode='lines',
#             line=dict(color=color_palette[i % len(color_palette)], width=1.5),
#             opacity=0.7  # Slightly increase opacity for clarity
#         ))

#     fig.update_layout(
#         title="Simulated Stock Price Paths",
#         xaxis_title="Time (Years)",
#         yaxis_title="Stock Price",
#         showlegend=False
#     )

#     # Display the plot in Streamlit
#     st.plotly_chart(fig)

#     # Histogram of payoffs
#     fig_payoff = go.Figure()
#     fig_payoff.add_trace(go.Histogram(x=payoffs, nbinsx=50, name="Payoff Distribution"))
#     fig_payoff.update_layout(
#         title="Distribution of Discounted Payoffs at Maturity",
#         xaxis_title="Discounted Payoff",
#         yaxis_title="Frequency",
#         showlegend=False
#     )

#     # Display payoff histogram
#     st.plotly_chart(fig_payoff)

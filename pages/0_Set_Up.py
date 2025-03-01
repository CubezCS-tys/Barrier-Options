import streamlit as st
import numpy as np

#############################################
# Title and Introduction
#############################################

st.title("Modeling Asset Price Paths")

st.header("Stochastic Processes")
st.write("""     
        
    Any variable whose value changes over time in an uncertain way is said to follow a
    stochastic process.
    """)
with st.expander("Discrete and Continuous variables and processes"):
    st.write("""  
    - **Discrete-Time Process:** The variable changes only at fixed points in time (e.g., every day, every month).
    
    - **Continuous-Time Process:** The variable can change at any moment in time.
    
    - **Continuous-Variable Process:** The variable can take any value within a certain range (often real numbers).
    
    - **Discrete-Variable Process:** The variable can only take on specific, discrete values (e.g., integers).
    
""")

st.markdown("""
    ### Markov property
    
    A Markov process is a stochastic process where predictions for the future depend solely on the current state, rendering past information irrelevant. In the context of stock prices, this means that only the current price matters for forecasting, and previous prices or the path taken do not affect future projections.
    """)
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import time

# Set a random seed for reproducibility
np.random.seed(40)

# Streamlit setup
st.write("### Live Simulation of Markov Property")
with st.expander("Simulation"):
    simulation_speed = st.slider("Simulation Speed (seconds per step):", 0.1, 2.0, 0.5)
    simulatem = st.button("Start Simulation", key="simulate_button_1")
    def markov_sim(simulation_speed):
        # Define parameters
        timesteps = 16  # Total number of timesteps to simulate

        # Placeholder for the plot
        placeholder = st.empty()

        # Initialise random walks
        previous_history_1 = [0]  # Start at position 0
        previous_history_2 = [0]

        # Simulation loop
        for step in range(1, timesteps + 1):
            # Generate the next step for each history
            next_step_1 = previous_history_1[-1] + np.random.choice([-1, 1])
            next_step_2 = previous_history_2[-1] + np.random.choice([-1, 1])

            # Append the new steps to the histories
            previous_history_1.append(next_step_1)
            previous_history_2.append(next_step_2)

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 5))

            # Plot histories
            x = np.arange(len(previous_history_1))  # Time steps
            ax.plot(x, previous_history_1, 'g-', linewidth=2, label="Previous History 1")
            ax.plot(x, previous_history_2, 'r-', linewidth=2, label="Previous History 2")

            # Annotate points
            ax.scatter([step], [previous_history_1[-1]], color="green", s=100)
            ax.scatter([step], [previous_history_2[-1]], color="red", s=100)

            # Annotate the plot
            ax.set_title(f"Markov Simulation: Step {step}")
            ax.set_xlim(0, timesteps + 2)
            ax.set_ylim(-timesteps, timesteps)
            ax.set_xlabel("Time")
            ax.set_ylabel("Position")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()

            # Render the plot in Streamlit
            placeholder.pyplot(fig)

            # Pause for the simulation speed
            time.sleep(simulation_speed)

        # After the simulation ends, display the possible future paths
        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot histories
        x = np.arange(len(previous_history_1))
        ax.plot(x, previous_history_1, 'g-', linewidth=2, label="Previous History 1")
        ax.plot(x, previous_history_2, 'r-', linewidth=2, label="Previous History 2")

        # Possible future paths for the first history
        future_steps = np.array([timesteps, timesteps + 1])
        future_path_1 = np.array([previous_history_1[-1], previous_history_1[-1] + 1])
        future_path_2 = np.array([previous_history_1[-1], previous_history_1[-1] - 1])
        ax.plot(future_steps, future_path_1, 'b--', linewidth=2, label="Future Path 1")
        ax.plot(future_steps, future_path_2, 'b--', linewidth=2, label="Future Path 2")

        # Annotate the final step
        ax.scatter([timesteps], [previous_history_1[-1]], color="green", s=100)
        ax.scatter([timesteps], [previous_history_2[-1]], color="red", s=100)

        # Annotate the plot
        ax.set_title("Markov Simulation: Final State with Future Paths")
        ax.set_xlim(0, timesteps + 2)
        ax.set_ylim(-timesteps, timesteps)
        ax.set_xlabel("Time")
        ax.set_ylabel("Position")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

        # Render the final plot with future paths in Streamlit
        placeholder.pyplot(fig)
    if simulatem:
        markov_sim(simulation_speed)
    
st.markdown("""

    
    The efficient market hypothesis (EMH) is the idea that financial markets process and reflect all available information in the prices of securities. There are three forms: 
    """)

with st.expander("Efficient Market Hypothesis"):
    st.write("""
    - **Weak Form:** Asserts that all past price information is fully reflected in current security prices, meaning historical price data cannot be used to achieve consistently above-average returns.

    - **Semi-Strong Form:** Stipulates that all publicly available information (not just price history, but also financial statements, news reports, etc.) is incorporated into current prices, making it difficult to benefit from newly released public information.

    - **Strong Form:** Proposes that all information, including both public and private (or insider) information, is fully reflected in security prices, implying that even insiders cannot consistently achieve above-average returns.
    """)

import streamlit as st

# Introduction
st.write("""
         ### Wiener Processes
A **Wiener process**, also known as standard Brownian motion, is a fundamental type of stochastic process that plays a key role in mathematical finance, physics, and other fields. It is a particular type of Markov process with the following properties:
""")

with st.expander("Properties of a Wiener Process"):
    st.write("""
    1. **Independent Increments**: The changes in the Wiener process over non-overlapping time intervals are independent of each other.
    2. **Normal Distribution**: Over a small time interval $\\Delta t$, the change in the Wiener process, $\\Delta z$, follows a normal distribution with:
    - Mean: $\\mathbb{E}[\\Delta z] = 0$,
    - Variance: $\\text{Var}[\\Delta z] = \\Delta t$.
    
    This can be written as:
    $$
    \\Delta z = \\epsilon \\sqrt{\\Delta t},
    $$
    where $\\epsilon \\sim \\phi(0, 1)$ is a standard normal random variable.

    3. **Continuous Paths**: The paths of a Wiener process are continuous, though they exhibit jaggedness or irregularity due to the random nature of the process.

    4. **Markov Property**: The process has no memory; the future evolution depends only on the current state and not on how the process arrived at that state.
    """)

# Formal Definition

with st.expander("Formal Definition"):
    st.write("""
    A Wiener process, $z(t)$, starting at $z(0) = 0$, satisfies the following:
    $$
    z(t) - z(0) \\sim \\phi(0, t),
    $$
    where:
    - The mean of the process is $0$,
    - The variance is equal to the elapsed time $t$,
    - The standard deviation is $\\sqrt{t}$.
    """)



st.write("""
         ### Generalized Wiener Process
The **Generalized Wiener Process** is an extension of the Wiener process, incorporating a drift rate and a volatility term. It can model processes where the mean and variance evolve over time, making it highly useful for financial modeling and other applications.
""")

# Definition
with st.expander("Definition"):
    st.write("""
    The generalized Wiener process for a variable $x$ can be expressed as:
    $$
    dx = a \\, dt + b \\, dz,
    $$
    where:
    - $a$ is the **drift rate** (mean change per unit time),
    - $b$ is the **volatility** (variance rate per unit time),
    - $dt$ is an infinitesimal increment in time,
    - $dz$ is a standard Wiener process.

    The drift term, $a \\, dt$, determines the expected rate of change, while the volatility term, $b \\, dz$, adds randomness or variability to the process.
    """)


# Variance and Distribution
st.subheader("Variance and Distribution")
st.write("""
For a time interval of length $\\Delta t$, the change in $x$ is given by:
$$
\\Delta x = a \\, \\Delta t + b \\, \\epsilon \\sqrt{\\Delta t},
$$
where $\\epsilon \\sim \\phi(0, 1)$ is a standard normal random variable. The change in $x$ has a normal distribution with:
- Mean: $a \\, \\Delta t$,
- Variance: $b^2 \\, \\Delta t$,
- Standard Deviation: $b \\sqrt{\\Delta t}$.

Over a period $T$, the change in $x$ follows:
$$
\\text{Mean: } aT, \\quad \\text{Variance: } b^2T, \\quad \\text{Standard Deviation: } b \\sqrt{T}.
$$
""")

# Example
with st.expander("Example: Cash Position of a Company"):
    st.write("""
    Consider a company's cash position, measured in thousands of dollars, following a generalized Wiener process with:
    - Drift rate $a = 20$ (per year),
    - Variance rate $b^2 = 900$ (per year).

    If the initial cash position is $50$:
    1. After 1 year, the cash position will follow:
    - Mean: $50 + 20 = 70$,
    - Standard Deviation: $\\sqrt{900} = 30$.
    - Distribution: $\\phi(70, 900)$.

    2. After 6 months ($T = 0.5$), the cash position will follow:
    - Mean: $50 + 20 \\times 0.5 = 60$,
    - Standard Deviation: $30 \\sqrt{0.5} \\approx 21.21$.
    - Distribution: $\\phi(60, 450)$.

    This demonstrates how uncertainty in the cash position increases with time, as measured by the standard deviation.
    """)

# Visual Illustration
st.subheader("Visualising the Generalised Wiener Process")
st.write("""
The generalized Wiener process is illustrated as a combination of:
1. A deterministic linear drift term ($a \\, dt$),
2. A random, jagged component ($b \\, dz$).

The figure below shows how different values of $a$ and $b$ affect the process:
""")

with st.expander("Generalised Wiener Process Simulation"):
    np.random.seed(42)
    # User inputs for simulation parameters
    #st.sidebar.header("Simulation Parameters")
    a = st.slider("Drift rate (a)", min_value=-5.0, max_value=5.0, step=0.1, value=0.5)
    b = st.slider("Volatility (b)", min_value=0.1, max_value=5.0, step=0.1, value=1.0)
    T = st.slider("Time horizon (T)", min_value=1, max_value=50, step=1, value=10)
    dt = st.slider("Time step size (dt)", min_value=0.01, max_value=1.0, step=0.01, value=0.1)
    simulate = st.button("Start Simulation", key="simulate_button_2")

    # Placeholder for the plot
    placeholder = st.empty()

    # Function to simulate the generalised Wiener process
    def simulate_generalised_wiener(a, b, T, dt):
        N = int(T / dt)  # Number of time steps
        times = np.linspace(0, T, N + 1)
        x = np.zeros(N + 1)  # Initialise the process (x[0] = 0 by default)
        w = np.zeros(N + 1)  # Wiener process (without drift)
        drift = a * times    # Linear drift rate

        for i in range(1, N + 1):
            dz = np.random.normal(0, np.sqrt(dt))  # Wiener increment
            dx = a * dt + b * dz  # Generalised Wiener process increment
            x[i] = x[i - 1] + dx  # Update the generalised Wiener process
            w[i] = w[i - 1] + b*dz  # Update the Wiener process

            # Plot the process dynamically
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(times[: i + 1], x[: i + 1], color="blue", label="Generalised Wiener Process")
            ax.plot(times[: i + 1], w[: i + 1], color="orange", label="Wiener Process (b * dz)")
            ax.plot(times[: i + 1], drift[: i + 1], color="green", linestyle="--", label="Linear Drift (a * t)")
            ax.set_title("Generalised Wiener Process Simulation", fontsize=16)
            ax.set_xlabel("Time", fontsize=14)
            ax.set_ylabel("Value of x", fontsize=14)
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)

            # Render the plot in Streamlit
            placeholder.pyplot(fig)
            time.sleep(0.05)  # Control the speed of the simulation

    # Run the simulation when the user clicks the button
    if simulate:
        simulate_generalised_wiener(a, b, T, dt)



# Introduction
st.write("""
         ### Itô Process
An **Itô process** is a type of stochastic process that extends the **generalised Wiener process**. In an Itô process, the drift rate and variance rate are not constant but depend on the current value of the variable and time. This makes it a flexible and powerful tool in modelling systems where the drift and volatility vary with the state or time.
""")

# Definition

with st.expander("Defintion"):
    st.write("""
    An Itô process is defined by the stochastic differential equation (SDE):
    $$
    dx = a(x, t) \\, dt + b(x, t) \\, dz,
    $$
    where:
    - $a(x, t)$ is the **drift rate**, which can change based on the current value of $x$ and time $t$.
    - $b(x, t)$ is the **volatility rate**, which also depends on $x$ and $t$.
    - $dt$ is an infinitesimal increment in time.
    - $dz$ is a standard Wiener process.

    This means that both the expected drift rate and variance rate of the process can vary dynamically.
    """)

# Discretised Form
st.subheader("Discretised Form of the Itô Process")
st.write("""
For a small time interval $\\Delta t$, the Itô process can be approximated as:
$$
\\Delta x = a(x, t) \\, \\Delta t + b(x, t) \\, \\epsilon \\sqrt{\\Delta t},
$$
where $\\epsilon \\sim \\phi(0, 1)$ is a standard normal random variable. This approximation assumes that the drift and variance rate remain constant during the small time interval.
""")

# Markov Property
st.subheader("Markov Property Applied")
st.write("""
The Itô process is **Markovian**, meaning the future behaviour of the process depends only on the current state $(x, t)$, not on its history. This is because the drift and variance rates are functions of the current state and time:
$$
a(x, t), \\quad b(x, t).
$$
Non-Markovian behaviour can occur if $a$ and $b$ depend on values of $x$ prior to time $t$.
""")


st.write("""
        #### Process for a Stock Price
This section discusses the stochastic process commonly assumed for modelling the price of a non-dividend-paying stock. It builds upon the generalised Wiener process but incorporates the unique characteristics of stock prices, such as proportional returns and volatility.
""")

# Key Assumptions
with st.expander("Expected Return ($\\mu$) and Volatility ($\\sigma$) as Percentages"):
    st.write("""
    - **Expected Return ($\\mu$)**:
    - The drift rate $\\mu$ represents the **expected percentage return** of the stock per unit time.
    - For example, if $\\mu = 15\\%$, it means the stock price is expected to increase by 15% per year on average.
    - **Volatility ($\\sigma$)**:
    - The volatility $\\sigma$ represents the **standard deviation of the stock's percentage return** per unit time.
    - For example, if $\\sigma = 30\\%$, it means the stock's returns typically deviate by 30% per year from the expected return.
    """)

# Deterministic Model (No Uncertainty)
st.subheader("Deterministic Model (No Uncertainty)")
st.write("""
If there is no uncertainty (i.e., no randomness), the change in the stock price, $S$, over a short interval $\\Delta t$ is given by:
$$
\\Delta S = \\mu S \\, \\Delta t,
$$
where $\\mu$ is the expected rate of return. Taking the limit as $\\Delta t \\to 0$, this becomes:
$$
\\frac{dS}{S} = \\mu \\, dt.
$$
Integrating this equation over the interval $[0, T]$ gives:
$$
S_T = S_0 e^{\\mu T},
$$
where $S_0$ is the stock price at time $t=0$. This shows that, without uncertainty, the stock price grows at a continuously compounded rate of $\\mu$ per unit of time.
""")

# Stochastic Model
st.subheader("Stochastic Model (Including Uncertainty)")
st.write("""
In reality, stock prices exhibit uncertainty due to market volatility. This leads to the following model:
$$
\\frac{dS}{S} = \\mu \\, dt + \\sigma \\, dz,
$$
or equivalently:
$$
dS = \\mu S \\, dt + \\sigma S \\, dz,
$$
where:
- $\\mu$ is the expected rate of return (drift rate),
- $\\sigma$ is the volatility of the stock price,
- $dz$ is a Wiener process (standard Brownian motion).

This model accounts for both the deterministic drift ($\\mu S \\, dt$) and the stochastic variability ($\\sigma S \\, dz$).
""")

# Discrete-Time Model
st.subheader("Discrete-Time Model")
st.write("""
The discrete-time approximation of the stochastic model is given by:
$$
\\frac{\\Delta S}{S} = \\mu \\, \\Delta t + \\sigma \\epsilon \\sqrt{\\Delta t},
$$
where:
- $\\epsilon \\sim \\phi(0, 1)$ is a standard normal random variable.
""")

st.write("""
Expanding this further, the change in the stock price $\\Delta S$ can be written as:
$$
\\Delta S = \\mu S \\, \\Delta t + \\sigma S \\epsilon \\sqrt{\\Delta t}.
$$
""")

# Distribution of Returns
with st.expander("Distribution of Returns"):
    st.write("""
    The returns $\\frac{\\Delta S}{S}$ over a short interval $\\Delta t$ are approximately normally distributed with:
    - Mean: $\\mu \\Delta t$,
    - Variance: $\\sigma^2 \\Delta t$.

    This is expressed as:
    $$
    \\frac{\\Delta S}{S} \\sim \\phi(\\mu \\Delta t, \\sigma^2 \\Delta t).
    $$
    """)


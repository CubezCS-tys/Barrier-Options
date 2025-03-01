# # import matplotlib.pyplot as plt
# # import streamlit as st

# # # Create a figure
# # fig, ax = plt.subplots(figsize=(10, 5))

# # # Plot paths
# # ax.plot([0, 1, 2, 3, 4], [2, 1, 2, 1, 0], 'r-', label="Previous History 2")  # Red line
# # ax.plot([0, 1, 2, 3, 4], [0, 1, 0, -1, 0], 'g-', label="Previous History 1")  # Green line
# # ax.plot([4, 5], [0, 1], 'b-', label="Possible Future Histories")  # Blue lines
# # ax.plot([4, 5], [0, -1], 'b-')

# # # Draw horizontal lines for positions
# # ax.axhline(y=1, color='black', linestyle='--', xmin=0, xmax=1)  # a+1
# # ax.axhline(y=0, color='black', linestyle='--', xmin=0, xmax=1)  # a
# # ax.axhline(y=-1, color='black', linestyle='--', xmin=0, xmax=1)  # a-1

# # # Annotate points
# # ax.scatter([0, 1, 2, 3], [0, 1, 2, 3], color='red')  # Red history
# # ax.scatter([0, 1, 2, 3], [0, -1, 0, 1], color='green')  # Green history
# # ax.scatter([5], [1], color='blue')  # Future history 1
# # ax.scatter([5], [-1], color='blue')  # Future history 2

# # # Add annotations
# # ax.annotate("position", (1.5, 2.1), fontsize=10, color="black")
# # ax.annotate("previous history 2", (0.5, 1.5), fontsize=10, color="red")
# # ax.annotate("previous history 1", (0.5, -0.5), fontsize=10, color="green")
# # ax.annotate("possible future histories", (2.5, 3.2), fontsize=10, color="blue")
# # ax.annotate("n-1", (0.3, -0.3), fontsize=10)
# # ax.annotate("n", (1.3, -0.3), fontsize=10)
# # ax.annotate("n+1", (2.3, -0.3), fontsize=10)

# # # Add explanatory text
# # ax.text(0.5, -1.5, "(For brevity, we will use commas instead of the set-theoretic symbol to denote conditioning on intersections of multiple events.)", fontsize=8, color="green")

# # # Set limits and labels
# # ax.set_xlim(-0.5, 5.5)
# # ax.set_ylim(-2, 4)
# # ax.set_xticks([0, 1, 2, 3, 4, 5])
# # ax.set_yticks([-1, 0, 1, 2, 3])
# # ax.set_xlabel("Time")
# # ax.set_ylabel("Position")

# # # Add legend
# # ax.legend()

# # # Display in Streamlit
# # st.pyplot(fig)

# import matplotlib.pyplot as plt
# import numpy as np
# import streamlit as st
# import time

# # Set a random seed for reproducibility
# np.random.seed(40)

# # Streamlit setup
# st.title("Live Simulation of Markov Property")
# st.write("A live simulation demonstrating paths and possible future histories.")

# # Define parameters
# timesteps = 16  # Total number of timesteps to simulate
# simulation_speed = st.slider("Simulation Speed (seconds per step):", 0.1, 2.0, 0.5)

# # Placeholder for the plot
# placeholder = st.empty()

# # Initialise random walks
# previous_history_1 = [0]  # Start at position 0
# previous_history_2 = [0]

# # Simulation loop
# for step in range(1, timesteps + 1):
#     # Generate the next step for each history
#     next_step_1 = previous_history_1[-1] + np.random.choice([-1, 1])
#     next_step_2 = previous_history_2[-1] + np.random.choice([-1, 1])

#     # Append the new steps to the histories
#     previous_history_1.append(next_step_1)
#     previous_history_2.append(next_step_2)

#     # Create figure
#     fig, ax = plt.subplots(figsize=(8, 5))

#     # Plot histories
#     x = np.arange(len(previous_history_1))  # Time steps
#     ax.plot(x, previous_history_1, 'g-', linewidth=2, label="Previous History 1")
#     ax.plot(x, previous_history_2, 'r-', linewidth=2, label="Previous History 2")

#     # Annotate points
#     ax.scatter([step], [previous_history_1[-1]], color="green", s=100)
#     ax.scatter([step], [previous_history_2[-1]], color="red", s=100)

#     # Annotate the plot
#     ax.set_title(f"Markov Simulation: Step {step}")
#     ax.set_xlim(0, timesteps + 2)
#     ax.set_ylim(-timesteps, timesteps)
#     ax.set_xlabel("Time")
#     ax.set_ylabel("Position")
#     ax.grid(True, linestyle='--', alpha=0.5)
#     ax.legend()

#     # Render the plot in Streamlit
#     placeholder.pyplot(fig)

#     # Pause for the simulation speed
#     time.sleep(simulation_speed)

# # After the simulation ends, display the possible future paths
# fig, ax = plt.subplots(figsize=(8, 5))

# # Plot histories
# x = np.arange(len(previous_history_1))
# ax.plot(x, previous_history_1, 'g-', linewidth=2, label="Previous History 1")
# ax.plot(x, previous_history_2, 'r-', linewidth=2, label="Previous History 2")

# # Possible future paths for the first history
# future_steps = np.array([timesteps, timesteps + 1])
# future_path_1 = np.array([previous_history_1[-1], previous_history_1[-1] + 1])
# future_path_2 = np.array([previous_history_1[-1], previous_history_1[-1] - 1])
# ax.plot(future_steps, future_path_1, 'b--', linewidth=2, label="Future Path 1")
# ax.plot(future_steps, future_path_2, 'b--', linewidth=2, label="Future Path 2")

# # Annotate the final step
# ax.scatter([timesteps], [previous_history_1[-1]], color="green", s=100)
# ax.scatter([timesteps], [previous_history_2[-1]], color="red", s=100)

# # Annotate the plot
# ax.set_title("Markov Simulation: Final State with Future Paths")
# ax.set_xlim(0, timesteps + 2)
# ax.set_ylim(-timesteps, timesteps)
# ax.set_xlabel("Time")
# ax.set_ylabel("Position")
# ax.grid(True, linestyle='--', alpha=0.5)
# ax.legend()

# # Render the final plot with future paths in Streamlit
# placeholder.pyplot(fig)

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import time

# Title
st.title("Simulating a Generalised Wiener Process")
np.random.seed(42)
# User inputs for simulation parameters
st.sidebar.header("Simulation Parameters")
a = st.sidebar.slider("Drift rate (a)", min_value=-5.0, max_value=5.0, step=0.1, value=0.5)
b = st.sidebar.slider("Volatility (b)", min_value=0.1, max_value=5.0, step=0.1, value=1.0)
T = st.sidebar.slider("Time horizon (T)", min_value=1, max_value=50, step=1, value=10)
dt = st.sidebar.slider("Time step size (dt)", min_value=0.01, max_value=1.0, step=0.01, value=0.1)
simulate = st.sidebar.button("Start Simulation")

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
        w[i] = w[i - 1] + dz  # Update the Wiener process

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

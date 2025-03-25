import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ----------------------------
# Simulation Function
# ----------------------------
def simulate_barrier_option(S, K, T, r, sigma, B, option_type, num_steps=100):
    dt = T / num_steps
    path = [S]
    barrier_activated = False

    # Simulate path based on Geometric Brownian Motion
    for _ in range(num_steps):
        Z = np.random.normal()
        next_price = path[-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
        path.append(next_price)

        # Barrier conditions
        if option_type == "Up-and-Out Call" and next_price >= B:
            return 0, path, True
        elif option_type == "Down-and-Out Call" and next_price <= B:
            return 0, path, True
        elif option_type == "Up-and-In Call" and next_price >= B:
            barrier_activated = True
        elif option_type == "Down-and-In Call" and next_price <= B:
            barrier_activated = True
        elif option_type == "Up-and-Out Put" and next_price >= B:
            return 0, path, True
        elif option_type == "Down-and-Out Put" and next_price <= B:
            return 0, path, True
        elif option_type == "Up-and-In Put" and next_price >= B:
            barrier_activated = True
        elif option_type == "Down-and-In Put" and next_price <= B:
            barrier_activated = True

    # Payoff calculation for options that did not breach "Out" barriers or activated "In" barriers
    if not barrier_activated and "Out" in option_type:
        payoff = max(path[-1] - K, 0) if "Call" in option_type else max(K - path[-1], 0)
    elif barrier_activated and "In" in option_type:
        payoff = max(path[-1] - K, 0) if "Call" in option_type else max(K - path[-1], 0)
    else:
        payoff = 0

    return payoff, path, barrier_activated

# ----------------------------
# Streamlit App Layout
# ----------------------------
st.set_page_config(page_title="Barrier Option Simulator", layout="wide")
st.title("Barrier Option Simulator")

# Inject CSS styles
box_style = """
<style>
.info-box {
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin: 10px; /* Adds space between boxes */
    border: 2px solid transparent; /* Default border */
}

.success {
    background-color: #90ee90;  /* Light green */
    border-color: #008000;      /* Green border */
}

.warning {
    background-color: #ffd700;  /* Gold */
    border-color: #ffc107;      /* Amber border */
}

.danger {
    background-color: #ffcccb;  /* Light red */
    border-color: #ff0000;      /* Red border */
}

.info {
    background-color: #add8e6;  /* Light blue */
    border-color: #17a2b8;      /* Blue border */
}
</style>
"""

st.markdown(box_style, unsafe_allow_html=True)

# Preload default values for each option type
default_params = {
    "Up-and-Out Call": {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "B": 120.0, "num_steps": 100},
    "Down-and-Out Call": {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "B": 80.0, "num_steps": 100},
    "Up-and-In Call": {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "B": 120.0, "num_steps": 100},
    "Down-and-In Call": {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "B": 80.0, "num_steps": 100},
    "Up-and-Out Put": {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "B": 120.0, "num_steps": 100},
    "Down-and-Out Put": {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "B": 80.0, "num_steps": 100},
    "Up-and-In Put": {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "B": 120.0, "num_steps": 100},
    "Down-and-In Put": {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "B": 80.0, "num_steps": 100}
}

# Option selection
option_type = st.selectbox(
    "Select Barrier Option Type",
    list(default_params.keys())
)

# Preload parameters based on selected option
params = default_params[option_type]
S = st.sidebar.number_input("Initial Stock Price (S)", min_value=0.01, value=params["S"], step=0.1)
K = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=params["K"], step=0.1)
T = st.sidebar.number_input("Time to Maturity (T in years)", min_value=0.01, value=params["T"], step=0.01)
r = st.sidebar.number_input("Risk-Free Interest Rate (r)", min_value=0.0, value=params["r"], step=0.01)
sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.01, value=params["sigma"], step=0.01)
B = st.sidebar.number_input("Barrier Level (B)", min_value=0.01, value=params["B"], step=0.1)
num_steps = st.sidebar.number_input("Number of Time Steps", min_value=10, value=params["num_steps"], step=10)

# Button to run the simulation
if st.button("Run Simulation"):
    # Run simulation
    payoff, path, barrier_breached = simulate_barrier_option(S, K, T, r, sigma, B, option_type, num_steps)

    # ----------------------------
    # Display Results in Color-Coded Boxes
    # ----------------------------
    # Create first row of boxes
    col1, col2, col3 = st.columns(3)

    # Box 1: Option Type
    with col1:
        st.markdown(
            f"""
            <div class="info-box info">
                <h3>Option Type</h3>
                <p>{option_type}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Box 2: Barrier Level
    with col2:
        st.markdown(
            f"""
            <div class="info-box info">
                <h3>Barrier Level (B)</h3>
                <p>{B}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    
    # Box 6: Time to Maturity
    with col3:
        st.markdown(
            f"""
            <div class="info-box info">
                <h3>Time to Maturity (T)</h3>
                <p>{T} years</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Create second row of boxes
    col4, col5, col6 = st.columns(3)

    # Box 4: Final Payoff
    with col4:
        if payoff > 0:
            box_color_payoff = "success"
        elif payoff == 0:
            box_color_payoff = "warning"
        else:
            box_color_payoff = "danger"

        st.markdown(
            f"""
            <div class="info-box {box_color_payoff}">
                <h3>Final Payoff</h3>
                <p>${payoff:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Box 5: Option Status
    with col5:
        if "Out" in option_type:
            if barrier_breached:
                status = "Option is Worthless"
                box_color_status = "danger"
            else:
                status = "Option Active"
                box_color_status = "success"
        elif "In" in option_type:
            if barrier_breached:
                status = "Option Activated"
                box_color_status = "success"
            else:
                status = "Option Not Activated"
                box_color_status = "danger"
        else:
            status = "N/A"
            box_color_status = "info"

        st.markdown(
            f"""
            <div class="info-box {box_color_status}">
                <h3>Option Status</h3>
                <p>{status}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Box 3: Barrier Breached
    if "Out" in option_type:
        if barrier_breached:
            box_color = "danger"
            breaching_text = "Yes"
        else:
            box_color = "success"
            breaching_text = "No"
    elif "In" in option_type:
        if barrier_breached:
            box_color = "success"
            breaching_text = "Yes"
        else:
            box_color = "danger"
            breaching_text = "No"
    else:
        box_color = "info"
        breaching_text = "N/A"

    with col6:
        st.markdown(
            f"""
            <div class="info-box {box_color}">
                <h3>Barrier Breached</h3>
                <p>{breaching_text}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ----------------------------
    # Plot the simulated path using Plotly for a sleek, interactive plot
    # ----------------------------
    fig = go.Figure()

    # Add stock price path
    fig.add_trace(go.Scatter(
        x=list(range(len(path))),
        y=path,
        mode="lines",
        name="Stock Price Path",
        line=dict(color="blue")
    ))

    # Add barrier level as a horizontal line
    fig.add_trace(go.Scatter(
        x=[0, len(path) - 1],
        y=[B, B],
        mode="lines",
        name="Barrier Level (B)",
        line=dict(color="red", dash="dash")
    ))

    # Customize layout for a clean, sleek look
    fig.update_layout(
        title=f"{option_type} Simulation",
        xaxis_title="Time Step",
        yaxis_title="Stock Price",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="plotly_white"
    )

    # Show payoff status on the chart as an annotation
    if barrier_breached and "Out" in option_type:
        fig.add_annotation(
            x=len(path) / 2,
            y=B + (B * 0.05),
            text="Barrier Breached - Option is Worthless",
            showarrow=False,
            font=dict(color="red", size=14)
        )
    elif not barrier_breached and "Out" in option_type:
        fig.add_annotation(
            x=len(path) - 1,
            y=path[-1],
            text=f"Payoff = ${payoff:.2f}",
            showarrow=True,
            arrowhead=2,
            font=dict(color="green", size=14)
        )
    elif barrier_breached and "In" in option_type:
        fig.add_annotation(
            x=len(path) - 1,
            y=path[-1],
            text=f"Payoff = ${payoff:.2f}",
            showarrow=True,
            arrowhead=2,
            font=dict(color="green", size=14)
        )
    elif not barrier_breached and "In" in option_type:
        fig.add_annotation(
            x=len(path) / 2,
            y=B + (B * 0.05),
            text="Barrier Not Breached - Option is Worthless",
            showarrow=False,
            font=dict(color="red", size=14)
        )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)
#################################################################################
#
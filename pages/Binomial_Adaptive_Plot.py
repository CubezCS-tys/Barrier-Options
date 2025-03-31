import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Function to plot adaptive binomial tree
def plot_adaptive_binomial_tree(S0, sigma, T, coarse_steps, fine_steps, barrier, fine_region):
    critical_region = (barrier*(1 - fine_region), barrier*(1 + fine_region))
    
    # Time intervals
    dt_coarse = T / coarse_steps
    dt_fine = T / fine_steps

    # Up and down factors for coarse and fine meshes
    u_coarse = np.exp(sigma * np.sqrt(dt_coarse))
    d_coarse = 1/u_coarse

    u_fine = np.exp(sigma * np.sqrt(dt_fine))
    d_fine = 1/u_fine

    nodes = []
    edges_x, edges_y = [], []

    def add_node(S, t, parent_x=None, parent_y=None):
        nodes.append((t, S))
        if parent_x is not None:
            edges_x.extend([parent_x, t, None])
            edges_y.extend([parent_y, S, None])

        # If not at maturity
        if t < T:
            is_fine = critical_region[0] <= S <= critical_region[1]
            dt, u, d = (dt_fine, u_fine, d_fine) if is_fine else (dt_coarse, u_coarse, d_coarse)

            add_node(S*u, t+dt, t, S)
            add_node(S*d, t+dt, t, S)

    # Start building the tree from the root node
    add_node(S0, 0)

    # Plotting the tree
    fig = go.Figure()

    # Edges
    fig.add_trace(go.Scatter(
        x=edges_x, y=edges_y,
        mode='lines',
        line=dict(color='gray', width=1),
        hoverinfo='none'
    ))

    # Nodes
    xs, ys = zip(*nodes)
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode='markers',
        marker=dict(color='blue', size=5),
        name='Nodes'
    ))

    # Barrier line
    fig.add_trace(go.Scatter(
        x=[0, T],
        y=[barrier, barrier],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Barrier'
    ))

    fig.update_layout(
        title="Adaptive Binomial Tree",
        xaxis_title="Time",
        yaxis_title="Asset Price",
        template="simple_white"
    )

    return fig

# Streamlit UI
st.title("Adaptive Binomial Tree Visualization")

col1, col2 = st.columns(2)

with col1:
    S0 = st.number_input("Initial Stock Price (S0)", 100.0)
    sigma = st.number_input("Volatility (σ)", 0.2)
    T = st.number_input("Time to maturity (T)", 1.0)

with col2:
    coarse_steps = st.number_input("Coarse Steps", value = 5, min_value= 0)
    fine_steps = st.number_input("Fine Steps", value = 20, min_value = 0)
    barrier = st.number_input("Barrier Level", 110.0)
    fine_region = st.slider("Fine region (%) around barrier", 0.01, 0.3, 0.1)

if st.button("Plot Adaptive Binomial Tree"):
    fig = plot_adaptive_binomial_tree(S0, sigma, T, coarse_steps, fine_steps, barrier, fine_region)
    st.plotly_chart(fig, use_container_width=True)

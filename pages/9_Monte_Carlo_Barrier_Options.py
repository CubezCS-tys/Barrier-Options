
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm



# -------------------------------------------------------------------
# 1) Analytical Barrier Option Pricing Functions (Provided Code)
# -------------------------------------------------------------------
def calc_d1(S0, K, r, q, sigma, T):
    return (np.log(S0 / K) + (r - q + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))

def calc_d2(S0, K, r, q, sigma, T):
    return calc_d1(S0, K, r, q, sigma, T) - sigma * np.sqrt(T)

def calc_c(S0, K, r, q, sigma, T):
    d1 = calc_d1(S0, K, r, q, sigma, T)
    d2 = calc_d2(S0, K, r, q, sigma, T)
    return (S0 * np.exp(-q*T)*norm.cdf(d1)
            - K * np.exp(-r*T)*norm.cdf(d2))

def calc_p(S0, K, r, q, sigma, T):
    d1 = calc_d1(S0, K, r, q, sigma, T)
    d2 = calc_d2(S0, K, r, q, sigma, T)
    return (K * np.exp(-r*T)*norm.cdf(-d2)
            - S0 * np.exp(-q*T)*norm.cdf(-d1))

def calc_lambda(r, q, sigma):
    return (r - q + 0.5 * sigma**2) / (sigma**2)

def calc_y(barrier, S0, K, T, sigma, r, q):
    lam = calc_lambda(r, q, sigma)
    return (np.log((barrier**2)/(S0*K)) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

def calc_x1(S0, barrier, T, sigma, r, q):
    lam = calc_lambda(r, q, sigma)
    return (np.log(S0/barrier) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

def calc_y1(S0, barrier, T, sigma, r, q):
    lam = calc_lambda(r, q, sigma)
    return (np.log(barrier/S0) / (sigma*np.sqrt(T))) + lam*sigma*np.sqrt(T)

def black_scholes(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "Call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # Put
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def barrier_option_price(S0, K, T, r, q, sigma, barrier, option_type):
    """
    Returns the closed-form price for a variety of knock-in/out barrier options.
    """
    x1 = calc_x1(S0, barrier, T, sigma, r, q)
    y1 = calc_y1(S0, barrier, T, sigma, r, q)
    c  = calc_c(S0, K, r, q, sigma, T)
    p  = calc_p(S0, K, r, q, sigma, T)
    lam= calc_lambda(r, q, sigma)
    y  = calc_y(barrier, S0, K, T, sigma, r, q)

    # Down-and-in Call
    if option_type == 'down-and-in call' and barrier <= K and S0 <= barrier:
        vanilla = black_scholes(S0, K, T, r, sigma, "Call")
        return vanilla
    elif option_type == 'down-and-in call' and barrier <= K:
        cdi = (S0 * np.exp(-q*T) * (barrier/S0)**(2*lam) * norm.cdf(y)
               - K * np.exp(-r*T) * (barrier/S0)**(2*lam - 2)
                 * norm.cdf(y - sigma*np.sqrt(T)))
        return cdi
    elif option_type == 'down-and-in call' and barrier >= K:
        term1 = S0*np.exp(-q*T)*norm.cdf(x1)
        term2 = K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
        term3 = S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(y1)
        term4 = K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)*norm.cdf(y1 - sigma*np.sqrt(T))
        cdo   = term1 - term2 - term3 + term4
        if cdo < 0:
            cdo = 0
        return c - cdo

    # Down-and-out Call
    elif option_type == 'down-and-out call' and barrier <= K:
        cdi = (S0 * np.exp(-q*T) * (barrier/S0)**(2*lam) * norm.cdf(y)
               - K * np.exp(-r*T) * (barrier/S0)**(2*lam - 2)
                 * norm.cdf(y - sigma*np.sqrt(T)))
        cdo = c - cdi
        return max(cdo, 0)
    elif option_type == 'down-and-out call' and barrier >= K:
        term1 = S0*np.exp(-q*T)*norm.cdf(x1)
        term2 = K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
        term3 = S0*np.exp(-q*T)*((barrier/S0)**(2*lam))*norm.cdf(y1)
        term4 = K*np.exp(-r*T)*((barrier/S0)**(2*lam - 2))*norm.cdf(y1 - sigma*np.sqrt(T))
        cdo   = term1 - term2 - term3 + term4
        return max(cdo, 0)

    # Up-and-in Call
    elif option_type == 'up-and-in call' and barrier > K:
        cui = (S0*np.exp(-q*T)*norm.cdf(x1)
               - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
               - S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*(norm.cdf(-y) - norm.cdf(-y1))
               + K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)*(norm.cdf(-y + sigma*np.sqrt(T))
                  - norm.cdf(-y1 + sigma*np.sqrt(T))))
        return cui
    elif option_type == 'up-and-in call' and barrier <= K:
        return c

    # Up-and-out Call
    elif option_type == 'up-and-out call' and barrier <= K:
        return 0.0
    elif option_type == 'up-and-out call' and barrier > K:
        cui = (S0*np.exp(-q*T)*norm.cdf(x1)
               - K*np.exp(-r*T)*norm.cdf(x1 - sigma*np.sqrt(T))
               - S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*(norm.cdf(-y) - norm.cdf(-y1))
               + K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)*(norm.cdf(-y + sigma*np.sqrt(T))
                  - norm.cdf(-y1 + sigma*np.sqrt(T))))
        cuo = c - cui
        return max(cuo, 0)

    # Up-and-in Put
    elif option_type == 'up-and-in put' and barrier >= K and barrier <= S0:
        return black_scholes(S0, K, T, r, sigma, "Put")
    elif option_type == 'up-and-in put' and barrier >= K:
        pui = (-S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(-y)
               + K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)*norm.cdf(-y + sigma*np.sqrt(T)))
        return pui
    elif option_type == 'up-and-in put' and barrier <= K:
        puo = (
            -S0*np.exp(-q*T)*norm.cdf(-x1)
            + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
            + S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(-y1)
            - K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)*norm.cdf(-y1 + sigma*np.sqrt(T))
        )
        if puo < 0:
            puo = 0
            return black_scholes(S0, K, T, r, sigma, "Put")
        else:
            return black_scholes(S0, K, T, r, sigma, "Put") - puo

    # Up-and-out Put
    elif option_type == 'up-and-out put' and barrier >= K:
        pui = (-S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(-y)
               + K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)*norm.cdf(-y + sigma*np.sqrt(T)))
        if pui > 0:
            return p - pui
        else:
            return p
    elif option_type == 'up-and-out put' and barrier <= K:
        puo = (
            -S0*np.exp(-q*T)*norm.cdf(-x1)
            + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
            + S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*norm.cdf(-y1)
            - K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)*norm.cdf(-y1 + sigma*np.sqrt(T))
        )
        return max(puo, 0)

    # Down-and-in Put
    elif option_type == 'down-and-in put' and barrier < K and S0 < barrier:
        return black_scholes(S0, K, T, r, sigma, "Put")
    elif option_type == 'down-and-in put' and barrier > K:
        return p
    elif option_type == 'down-and-in put' and barrier < K:
        pdi = (
            -S0*np.exp(-q*T)*norm.cdf(-x1)
            + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
            + S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*(norm.cdf(y) - norm.cdf(y1))
            - K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)*(norm.cdf(y - sigma*np.sqrt(T))
               - norm.cdf(y1 - sigma*np.sqrt(T)))
        )
        return pdi

    # Down-and-out Put
    elif option_type == 'down-and-out put' and barrier > K:
        return 0
    elif option_type == 'down-and-out put' and barrier < K:
        pdi = (
            -S0*np.exp(-q*T)*norm.cdf(-x1)
            + K*np.exp(-r*T)*norm.cdf(-x1 + sigma*np.sqrt(T))
            + S0*np.exp(-q*T)*(barrier/S0)**(2*lam)*(norm.cdf(y) - norm.cdf(y1))
            - K*np.exp(-r*T)*(barrier/S0)**(2*lam - 2)*(norm.cdf(y - sigma*np.sqrt(T))
               - norm.cdf(y1 - sigma*np.sqrt(T)))
        )
        pdo = p - pdi
        return max(pdo, 0)

    return None

# -------------------------------------------------------------------
# 2) Monte Carlo Barrier Option Pricing (with variance reduction)
# -------------------------------------------------------------------
def monte_carlo_barrier_option_combined(S0, K, H, T, r, q, sigma, M, N, option_type, rebate=0,
                                        use_antithetic=False, use_control_variate=False):
    """
    MC pricing for barrier options, returning:
      - barrier_price_cv (final price if control variate is used; else same as barrier_price_mc)
      - barrier_price_mc (raw MC estimate of the barrier option)
      - vanilla_mc (MC estimate of a plain call or put payoff, used for control variate)
    """
    dt = T / M
    disc_factor = np.exp(-r * T)
    
    is_call = 'call' in option_type.lower()
    is_up   = 'up'   in option_type.lower()
    is_in   = 'in'   in option_type.lower()
    
    payoffs         = np.zeros(N)
    vanilla_payoffs = np.zeros(N)  # used for control variate

    for i in range(N):
        S_plus = S0
        S_minus = S0
        breach_plus  = False
        breach_minus = False
        for t in range(M):
            z = np.random.standard_normal()
            S_plus *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
            if not breach_plus:
                if (is_up and S_plus >= H) or ((not is_up) and S_plus <= H):
                    breach_plus = True
            
            if use_antithetic:
                z_antithetic = -z
                S_minus *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z_antithetic)
                if not breach_minus:
                    if (is_up and S_minus >= H) or ((not is_up) and S_minus <= H):
                        breach_minus = True
            else:
                S_minus = S_plus
                breach_minus = breach_plus
        
        # Barrier payoff
        if is_in:
            payoff_plus = (max(S_plus - K, 0) if is_call and breach_plus 
                           else max(K - S_plus, 0) if (not is_call) and breach_plus else rebate)
            payoff_minus = (max(S_minus - K, 0) if is_call and breach_minus 
                            else max(K - S_minus, 0) if (not is_call) and breach_minus else rebate)
        else:
            payoff_plus = (max(S_plus - K, 0) if is_call and (not breach_plus)
                           else max(K - S_plus, 0) if (not is_call) and (not breach_plus) else rebate)
            payoff_minus = (max(S_minus - K, 0) if is_call and (not breach_minus)
                            else max(K - S_minus, 0) if (not is_call) and (not breach_minus) else rebate)
        
        if use_antithetic:
            payoffs[i] = 0.5*(payoff_plus + payoff_minus)
        else:
            payoffs[i] = payoff_plus
        
        # For control variate: plain vanilla payoff
        vanilla_plus  = max(S_plus - K, 0) if is_call else max(K - S_plus, 0)
        vanilla_minus = max(S_minus - K, 0) if is_call else max(K - S_minus, 0)
        if use_antithetic:
            vanilla_payoffs[i] = 0.5*(vanilla_plus + vanilla_minus)
        else:
            vanilla_payoffs[i] = vanilla_plus

    barrier_price_mc = disc_factor*np.mean(payoffs)
    vanilla_mc       = disc_factor*np.mean(vanilla_payoffs)
    
    if use_control_variate:
        if is_call:
            vanilla_analytic = black_scholes(S0, K, T, r, sigma, "Call")
        else:
            vanilla_analytic = black_scholes(S0, K, T, r, sigma, "Put")
        barrier_price_cv = barrier_price_mc - vanilla_mc + vanilla_analytic
    else:
        barrier_price_cv = barrier_price_mc
    
    return barrier_price_cv, barrier_price_mc, vanilla_mc

# -------------------------------------------------------------------
# 3) Sample Path Simulation (for plotting)
# -------------------------------------------------------------------
def simulate_paths(S0, K, H, T, r, q, sigma, M, N, option_type, rebate, use_antithetic):
    dt = T / M
    time_grid = np.linspace(0, T, M+1)
    #num_paths = min(500, 10000)
    num_paths = N
    sample_paths  = []
    payoff_samples= []
    sample_status = []
    
    is_call = 'call' in option_type.lower()
    is_up   = 'up'   in option_type.lower()
    is_in   = 'in'   in option_type.lower()
    
    for i in range(num_paths):
        S_path  = [S0]
        S_val   = S0
        breached= False
        for t in range(M):
            z = np.random.standard_normal()
            S_val *= np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
            S_path.append(S_val)
            if not breached:
                if (is_up and S_val >= H) or ((not is_up) and S_val <= H):
                    breached = True
        sample_paths.append(S_path)
        if is_in:
            payoff = (max(S_val - K, 0) if is_call and breached 
                      else max(K - S_val, 0) if (not is_call) and breached else rebate)
        else:
            payoff = (max(S_val - K, 0) if is_call and (not breached)
                      else max(K - S_val, 0) if (not is_call) and (not breached) else rebate)
        payoff_samples.append(payoff)
        sample_status.append(breached if is_in else (not breached))
    
    return time_grid, sample_paths, payoff_samples, sample_status

# -------------------------------------------------------------------
# 4) Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="Barrier Option Pricing (MC + Analytical)", layout="wide")
st.title("Barrier Option Pricing with Monte Carlo and Analytical Comparison")
st.write("""
This app prices barrier options using Monte Carlo simulation (with optional variance reduction) and 
compares the results to an **analytical** barrier pricing formula.
""")

# Sidebar: Choose MC methods
st.sidebar.header("Monte Carlo Methods")
use_standard   = st.sidebar.checkbox("Standard MC", value=True)
use_antithetic = st.sidebar.checkbox("Antithetic Variates", value=False)
use_control    = st.sidebar.checkbox("Control Variate", value=False)
if not (use_standard or use_antithetic or use_control):
    use_standard = True
methods_to_run = {}
if use_standard:
    methods_to_run["Standard MC"] = (False, False)
if use_antithetic:
    methods_to_run["Antithetic MC"] = (True,  False)
if use_control:
    methods_to_run["Control Variate MC"] = (False, True)
if use_antithetic and use_control:
    methods_to_run["Combined MC"] = (True, True)

# Sidebar: Barrier Option Parameters
st.sidebar.header("Barrier Option Parameters")
option_list = [
    'Up-and-Out Call', 'Down-and-Out Call', 'Up-and-Out Put', 'Down-and-Out Put',
    'Up-and-In Call', 'Down-and-In Call', 'Up-and-In Put', 'Down-and-In Put'
]
option_type = st.sidebar.selectbox("Barrier Option Type", option_list)
S0    = st.sidebar.number_input("Initial Stock Price (S0)", min_value=0.01, value=100.0)
K     = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=100.0)
H     = st.sidebar.number_input("Barrier Level (H)", min_value=0.01, value=110.0)
T     = st.sidebar.number_input("Time to Maturity (T)", min_value=0.01, value=1.0, step=0.1)
r     = st.sidebar.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.05, step=0.01, format="%.4f")
q     = st.sidebar.number_input("Dividend Yield (q)", min_value=0.0, value=0.02, step=0.01, format="%.4f")
sigma = st.sidebar.number_input("Volatility (σ in %)", min_value=0.01, value=20.0, step=0.1, format="%.2f")/100
M     = st.sidebar.number_input("Number of Time Steps (M)", min_value=1, value=100)
N     = st.sidebar.number_input("Number of Simulation Paths (N)", min_value=1, value=10000, step=100)
rebate= st.sidebar.number_input("Rebate", min_value=0.0, value=0.0, step=1.0)
S_min = st.sidebar.number_input("Sensitivity - Minimum Stock Price", value=80.0)
S_max = st.sidebar.number_input("Sensitivity - Maximum Stock Price", value=120.0)
num_points = st.sidebar.number_input("Sensitivity - Number of Points", value=20, step=1)

option_map = {
    'Up-and-Out Call':    'up-and-out call',
    'Down-and-Out Call':  'down-and-out call',
    'Up-and-Out Put':     'up-and-out put',
    'Down-and-Out Put':   'down-and-out put',
    'Up-and-In Call':     'up-and-in call',
    'Down-and-In Call':   'down-and-in call',
    'Up-and-In Put':      'up-and-in put',
    'Down-and-In Put':    'down-and-in put'
}
option_type_snippet = option_map[option_type]

# Custom CSS for info boxes
st.markdown(
    """
    <style>
    .info-box {
        background-color: #f9f9f9;
        border-left: 5px solid #007ACC;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        text-align: center;
    }
    .info-box h4 {
        margin: 0;
        color: #007ACC;
    }
    .info-box p {
        margin: 0.5rem 0 0;
        font-size: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True
)
def create_info_box(title, value):
    return f"<div class='info-box'><h4>{title}</h4><p>{value}</p></div>"

# # Compute analytical barrier price once
barrier_analytic = barrier_option_price(S0, K, T, r, q, sigma, H, option_type_snippet)

if st.sidebar.button("Calculate Option Price"):
    # Create tabs for each MC approach
    tab_labels = list(methods_to_run.keys())
    tabs = st.tabs(tab_labels)
    
    for idx, method_name in enumerate(tab_labels):
        flag_antithetic, flag_control = methods_to_run[method_name]
        barrier_cv, barrier_mc, vanilla_mc = monte_carlo_barrier_option_combined(
            S0, K, H, T, r, q, sigma, int(M), int(N), option_type, rebate,
            use_antithetic=flag_antithetic, use_control_variate=flag_control
        )
        mc_price_final = barrier_cv if flag_control else barrier_mc
        
        # Simulate sample paths for the tab's chosen method
        time_grid, sample_paths, payoff_samples, sample_status = simulate_paths(
            S0, K, H, T, r, q, sigma, int(M), int(N),option_type, rebate, flag_antithetic
        )
        
        with tabs[idx]:
            st.markdown("### Results")
            
            # Show MC vs. Analytical prices side by side
            colA, colB = st.columns(2)
            with colA:
                if mc_price_final is not None:
                    st.markdown(create_info_box(f"{method_name} Price", f"${mc_price_final:.4f}"),
                                unsafe_allow_html=True)
                else:
                    st.markdown(create_info_box("Error", "MC returned None"),
                                unsafe_allow_html=True)
            with colB:
                if barrier_analytic is not None:
                    st.markdown(create_info_box("Analytical Barrier Price", f"${barrier_analytic:.4f}"),
                                unsafe_allow_html=True)
                else:
                    st.markdown(create_info_box("Analytical Price", "None / Invalid"),
                                unsafe_allow_html=True)
            
            # Plot simulated paths
            fig_paths = go.Figure()
            for i in range(len(sample_paths)):
                color = "rgba(0,255,0,0.6)" if sample_status[i] else "rgba(255,0,0,0.6)"
                fig_paths.add_trace(go.Scatter(
                    x=time_grid, y=sample_paths[i],
                    mode='lines',
                    line=dict(color=color, width=1),
                    opacity=0.5
                ))
            fig_paths.add_trace(go.Scatter(
                x=[0, T], y=[H, H],
                mode='lines',
                line=dict(color='blue', dash='dash'),
                name='Barrier'
            ))
            fig_paths.update_layout(
                title="Simulated Stock Price Paths",
                xaxis_title='Time (Years)',
                yaxis_title='Stock Price',
                showlegend=False
            )
            
            # Histogram of final payoffs
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=payoff_samples, nbinsx=30, marker_color='teal'))
            fig_hist.update_layout(
                title="Distribution of Final Payoffs",
                xaxis_title="Payoff",
                yaxis_title="Frequency"
            )
            
            col_left, col_right = st.columns(2)
            with col_left:
                st.plotly_chart(fig_paths, use_container_width=True)
            with col_right:
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # ---------------------------
            # SENSITIVITY ANALYSIS FOR THIS METHOD
            # ---------------------------
            st.subheader("Sensitivity Analysis: Price vs. Stock Price")
            S_range = np.linspace(S_min, S_max, int(num_points))
            analytical_prices = []
            mc_prices = []
            errors = []
            
            for S_val in S_range:
                # Analytical
                analytic_price = barrier_option_price(S_val, K, T, r, q, sigma, H, option_type_snippet)
                
                # Monte Carlo for THIS method's settings
                barrier_cv_sens, barrier_mc_sens, _ = monte_carlo_barrier_option_combined(
                    S_val, K, H, T, r, q, sigma, int(M), int(N),
                    option_type_snippet, rebate,
                    use_antithetic=flag_antithetic,
                    use_control_variate=flag_control
                )
                mc_price_sens = barrier_cv_sens if flag_control else barrier_mc_sens
                
                analytical_prices.append(analytic_price)
                mc_prices.append(mc_price_sens)
                errors.append(mc_price_sens - analytic_price)
            
            # Plot MC vs. Analytical
            fig_prices = go.Figure()
            fig_prices.add_trace(go.Scatter(
                x=S_range, y=analytical_prices,
                mode='lines+markers',
                name="Analytical Price",
                line=dict(color='blue', width=2)
            ))
            fig_prices.add_trace(go.Scatter(
                x=S_range, y=mc_prices,
                mode='lines+markers',
                name="Monte Carlo Price",
                line=dict(color='orange', width=2)
            ))
            fig_prices.update_layout(
                title="Barrier Option Price vs. Stock Price",
                xaxis_title="Stock Price",
                yaxis_title="Option Price",
                legend=dict(x=0, y=1)
            )
            
            # Plot Error
            fig_errors = go.Figure()
            fig_errors.add_trace(go.Scatter(
                x=S_range, y=errors,
                mode='lines+markers',
                name="Error (MC - Analytical)",
                line=dict(color='red', width=2)
            ))
            fig_errors.update_layout(
                title="Pricing Error vs. Stock Price",
                xaxis_title="Stock Price",
                yaxis_title="Error"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_prices, use_container_width=True)
            with col2:
                st.plotly_chart(fig_errors, use_container_width=True)

else:
    st.info("Enter parameters and click 'Calculate Option Price' to see results.")


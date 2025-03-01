import streamlit as st
import numpy as np

###############################################################################
# Trinomial Tree Pricer
###############################################################################
def build_stock_tree(S0, r, sigma, T, N):
    """
    Build and return a 2D array of underlying asset prices for a trinomial tree.
    Each row i corresponds to time step i, and each column j corresponds
    to the 'node index' at that step (from -i to +i in steps of 1, but we'll
    store them in 0..2i for convenience).
    
    S0    : initial stock price
    r     : risk-free interest rate
    sigma : volatility
    T     : time to maturity
    N     : number of time steps
    """
    dt = T / N
    # Common “standard” choice for up/down factors in a trinomial tree:
    u = np.exp(sigma * np.sqrt(3 * dt))
    d = 1.0 / u
    m = 1.0  # "middle" move is effectively S * m

    # Create a 2D array (list of lists) for stock prices
    # At step i, we have (2*i + 1) nodes in a recombining trinomial.
    stock_tree = []
    stock_tree.append([S0])  # at time 0, only one node

    for i in range(1, N + 1):
        level_prices = []
        # The middle index at step i is i (0..2i), so:
        for j in range(2 * i + 1):
            # The “shift” from the middle node is j - i
            k = j - i
            # Price = S0 * u^(#ups) * d^(#downs)
            # #ups - #downs = k, so #ups = i + k / 2, #downs = i - k / 2 (in a binomial sense)
            # but simpler is to say: each step can be up, down, or mid.
            # We can interpret k>0 as net "ups" and k<0 as net "downs".
            # For a purely multiplicative approach:
            price = S0 * (u ** max(k, 0)) * (d ** max(-k, 0))
            level_prices.append(price)
        stock_tree.append(level_prices)

    return stock_tree

def build_barrier_status_tree(stock_tree, barrier_type, barrier_level, N):
    """
    Given the stock tree, determine whether each node is 'active' (has not
    knocked out) or 'inactive' (knocked out), or if we handle knock-in.

    This function returns a 2D list (same shape as stock_tree) of booleans
    indicating whether the option at that node is "alive" (True) or "dead" (False).
    
    For 'knock-out' barriers, once the barrier is touched, we mark that path as dead.
    For 'knock-in' barriers, we do the opposite (the option only becomes alive if the
    barrier is touched).  Here we assume discrete monitoring at each time step.

    barrier_type can be one of:
      - 'up-and-out'
      - 'down-and-out'
      - 'up-and-in'
      - 'down-and-in'
      or None if no barrier is used.

    barrier_level : the barrier price
    N             : number of time steps
    """
    # If no barrier, everything is alive
    if not barrier_type:
        return [[True for _ in row] for row in stock_tree]

    # We'll do a forward pass, checking barrier at each node:
    # For knock-out, once triggered -> dead. For knock-in, once triggered -> alive.
    barrier_tree = []
    barrier_tree.append([False] * len(stock_tree[0]))  # time=0 row
    # Initialization at time 0:
    S0 = stock_tree[0][0]
    if barrier_type == 'up-and-out':
        # If S0 >= barrier_level, it's knocked out immediately
        barrier_tree[0][0] = (S0 < barrier_level)
    elif barrier_type == 'down-and-out':
        # If S0 <= barrier_level, it's knocked out immediately
        barrier_tree[0][0] = (S0 > barrier_level)
    elif barrier_type == 'up-and-in':
        # If S0 >= barrier_level, it's already in
        barrier_tree[0][0] = (S0 >= barrier_level)
    elif barrier_type == 'down-and-in':
        # If S0 <= barrier_level, it's already in
        barrier_tree[0][0] = (S0 <= barrier_level)

    for i in range(1, N + 1):
        row_alive = []
        for j in range(len(stock_tree[i])):
            # We can come from j, j-1, j+1 in the previous row (in principle),
            # but for simplicity in a recombining tree, let's check direct adjacency:
            # In practice, you'd carefully map indices. We'll do a broad approach:
            # parent indices in previous row could be j, j-1, j+1, if valid.
            parents = []
            for pj in [j-1, j, j+1]:
                if 0 <= pj < len(stock_tree[i-1]):
                    parents.append(pj)
            # The node is alive if at least one parent path is alive
            # AND we haven't triggered a knock-out, or we have triggered a knock-in, etc.
            current_price = stock_tree[i][j]

            # Check if this node triggers barrier:
            if barrier_type in ['up-and-out', 'down-and-out']:
                # We want to remain alive only if none of the parents were knocked out
                # and we don't cross the barrier now
                was_alive = any(barrier_tree[i-1][pj] for pj in parents)
                if barrier_type == 'up-and-out':
                    # Knock out if current_price >= barrier
                    still_alive = was_alive and (current_price < barrier_level)
                else:
                    # down-and-out
                    still_alive = was_alive and (current_price > barrier_level)

            else:
                # 'up-and-in' or 'down-and-in'
                # We are alive if we have triggered the barrier at any point
                was_alive = any(barrier_tree[i-1][pj] for pj in parents)
                if barrier_type == 'up-and-in':
                    # We "become alive" if we cross barrier
                    # If not crossed barrier yet, remain not alive
                    # If at this node price >= barrier, then we are definitely in
                    # Or if the parents were already in.
                    triggered = (current_price >= barrier_level) or was_alive
                    still_alive = triggered
                else:
                    # down-and-in
                    triggered = (current_price <= barrier_level) or was_alive
                    still_alive = triggered

            row_alive.append(still_alive)
        barrier_tree.append(row_alive)

    return barrier_tree

def trinomial_option_price(S0, K, r, sigma, T, N,
                           option_type='call',
                           option_style='European',
                           barrier_type=None,
                           barrier_level=None):
    """
    Price a (European/American) call or put option via a trinomial tree.
    Can also handle simple barrier logic (knock-in or knock-out) if barrier_type
    is given (and barrier_level is set).
    
    S0           : initial underlying price
    K            : strike
    r            : risk-free rate
    sigma        : volatility
    T            : time to maturity
    N            : number of time steps
    option_type  : 'call' or 'put'
    option_style : 'European' or 'American'
    barrier_type : None, 'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'
    barrier_level: numeric barrier
    """
    dt = T / N
    # Risk-neutral discount factor
    discount = np.exp(-r * dt)
    
    # Up, down, middle factors
    u = np.exp(sigma * np.sqrt(3 * dt))
    d = 1.0 / u
    m = 1.0
    
    # One common set of risk-neutral probabilities that match the first two moments:
    # (There are various equivalent formulas in literature. This is a standard one.)
    # Here is one approach (for simplicity):
    pu = 1/6 + ( (r - 0.5*sigma**2)*np.sqrt(dt) / (2*sigma*np.sqrt(3)) )
    pm = 2/3
    pd = 1 - pu - pm
    # You might want to ensure pu, pm, pd are valid (>=0) under your chosen parameters.
    
    # Build the stock price tree
    stock_tree = build_stock_tree(S0, r, sigma, T, N)
    
    # Build the barrier status tree (True=alive, False=dead),
    # for knock-out or knock-in logic
    barrier_tree = build_barrier_status_tree(stock_tree, barrier_type, barrier_level, N)
    
    # Initialize option value at maturity
    # For a barrier option, if the node is not "alive" (for knock-out),
    # the payoff is 0.  For knock-in, if the node is not "alive" at maturity,
    # the payoff is 0.
    payoff_tree = []
    for i in range(N + 1):
        payoff_tree.append([0.0]*(2*i + 1))
    
    # Maturity payoffs
    for j in range(len(stock_tree[N])):
        S_T = stock_tree[N][j]
        alive = barrier_tree[N][j]
        if alive:
            if option_type == 'call':
                payoff_tree[N][j] = max(S_T - K, 0.0)
            else:  # put
                payoff_tree[N][j] = max(K - S_T, 0.0)
        else:
            payoff_tree[N][j] = 0.0  # not alive => no payoff
    
    # Backward induction
    for i in range(N-1, -1, -1):
        for j in range(len(stock_tree[i])):
            # Only compute if barrier_tree[i][j] is alive
            if not barrier_tree[i][j]:
                payoff_tree[i][j] = 0.0
                continue
            
            # Expected value from next step
            # child indices in next row: j, j+1, j+2  (because each node i, j
            # transitions to i+1, j, j+1, j+2 in a recombining tri-tree layout).
            # But we have to be careful with indexing. The middle child is j+1, up child is j+2, down child is j.
            # We'll do it carefully:
            V_u = payoff_tree[i+1][j+2] if (j+2 < len(payoff_tree[i+1])) else 0.0
            V_m = payoff_tree[i+1][j+1] if (j+1 < len(payoff_tree[i+1])) else 0.0
            V_d = payoff_tree[i+1][j]   if (j < len(payoff_tree[i+1]))     else 0.0
            
            # Also must ensure that the barrier_tree is alive for the child nodes
            # If the child node is dead, that path payoff is 0
            A_u = barrier_tree[i+1][j+2] if (j+2 < len(barrier_tree[i+1])) else False
            A_m = barrier_tree[i+1][j+1] if (j+1 < len(barrier_tree[i+1])) else False
            A_d = barrier_tree[i+1][j]   if (j < len(barrier_tree[i+1]))   else False
            
            V_expected = pu*(V_u if A_u else 0.0) \
                         + pm*(V_m if A_m else 0.0) \
                         + pd*(V_d if A_d else 0.0)
            
            # Discount back
            continuation_value = discount * V_expected
            
            if option_style == 'American':
                # Check early exercise
                S_ij = stock_tree[i][j]
                if option_type == 'call':
                    exercise_value = max(S_ij - K, 0.0)
                else:
                    exercise_value = max(K - S_ij, 0.0)
                payoff_tree[i][j] = max(continuation_value, exercise_value)
            else:
                payoff_tree[i][j] = continuation_value
    
    # The option price is at the root
    return payoff_tree[0][0]

###############################################################################
# Streamlit UI
###############################################################################
def main():
    st.title("Trinomial Tree Option Pricer")

    st.sidebar.header("Model Parameters")
    S0 = st.sidebar.number_input("Initial Underlying Price (S0)", value=100.0)
    K  = st.sidebar.number_input("Strike (K)", value=100.0)
    r  = st.sidebar.number_input("Risk-free Rate (r)", value=0.05, format="%.5f")
    sigma = st.sidebar.number_input("Volatility (sigma)", value=0.2, format="%.5f")
    T  = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0)
    N  = st.sidebar.number_input("Number of Steps (N)", min_value=1, value=50)

    st.sidebar.header("Option Type & Style")
    option_type = st.sidebar.selectbox("Option Type", ["call", "put"])
    option_style = st.sidebar.selectbox("Option Style", ["European", "American"])

    st.sidebar.header("Barrier (Optional)")
    barrier_features = [None, "up-and-out", "down-and-out", "up-and-in", "down-and-in"]
    barrier_type = st.sidebar.selectbox("Barrier Type", barrier_features, index=0)
    barrier_level = None
    if barrier_type is not None and barrier_type != "None":
        barrier_level = st.sidebar.number_input("Barrier Level", value=120.0)

    if st.button("Compute Option Price"):
        price = trinomial_option_price(
            S0=S0,
            K=K,
            r=r,
            sigma=sigma,
            T=T,
            N=int(N),
            option_type=option_type,
            option_style=option_style,
            barrier_type=(barrier_type if barrier_type != "None" else None),
            barrier_level=barrier_level
        )
        st.write(f"### Option Price = {price:0.4f}")

if __name__ == "__main__":
    main()

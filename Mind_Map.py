import networkx as nx
import matplotlib.pyplot as plt

# Define the project structure as a nested dictionary
structure = {
    '1': {
        'title': 'Introduction',
        'subsections': {
            '1.1': {'title': 'Background and Motivation'},
            '1.2': {'title': 'Objectives of the Study'},
            '1.3': {'title': 'Structure of the Dissertation'}
        }
    },
    '2': {
        'title': 'Literature Review',
        'subsections': {
            '2.1': {
                'title': 'Fundamentals of Option Pricing',
                'subsections': {
                    '2.1.1': {'title': 'Black-Scholes Model'},
                    '2.1.2': {'title': 'Assumptions and Limitations'}
                }
            },
            '2.2': {
                'title': 'Barrier Options',
                'subsections': {
                    '2.2.1': {'title': 'Definitions and Types'},
                    '2.2.2': {'title': 'Existing Analytical Solutions'}
                }
            }
            # Add more subsections as needed
        }
    }
    # Add more sections (3 to 10) as needed
}

# Create a directed graph
G = nx.DiGraph()

# Function to add nodes and edges
def add_nodes_edges(parent_id, parent_title, subsections):
    for key, value in subsections.items():
        node_id = key
        node_title = value['title']
        G.add_node(node_id, title=node_title)
        G.add_edge(parent_id, node_id)
        if 'subsections' in value:
            add_nodes_edges(node_id, node_title, value['subsections'])

# Add root node
root_id = 'Title'
root_title = 'Pricing of Barrier Options:\nAnalytical and Numerical Approaches'
G.add_node(root_id, title=root_title)

# Build the graph
add_nodes_edges(root_id, root_title, structure)

# Position nodes using a hierarchical layout
pos = nx.spring_layout(G, k=0.5, iterations=100)

# Extract labels
labels = {node: G.nodes[node]['title'] for node in G.nodes()}

# Draw the graph
plt.figure(figsize=(12, 8))
nx.draw(G, pos, labels=labels, with_labels=True, node_size=2000, node_color='lightblue', font_size=8, arrows=True)
plt.title('Mind Map of Final Year Project')
plt.show()

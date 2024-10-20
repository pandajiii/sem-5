#Install netwokx in terminal:
#syntax:> pip install networkx

import networkx as nx

G = nx.DiGraph()

G.add_edges_from([
    (0, 2),  
    (1, 0),  
    (2, 1),  
    (2, 3)   
])

# PageRank Algorithm Implementation
def pagerank(graph, alpha=0.85, num_iterations=100):
    # Calculate PageRank using the networkx built-in function
    return nx.pagerank(graph, alpha=alpha, max_iter=num_iterations)

# HITS Algorithm Implementation
def hits(graph, num_iterations=100):
    return nx.hits(graph, max_iter=num_iterations)


pagerank_scores = pagerank(G)

hubs, authorities = hits(G)

# Print the results
print("PageRank Scores:")
for node, score in pagerank_scores.items():
    print(f"Page {node}: {score:.4f}")

print("\nHITS Algorithm Results:")
print("Hubs Scores:")
for node, score in hubs.items():
    print(f"Page {node}: {score:.4f}")

print("Authorities Scores:")
for node, score in authorities.items():
    print(f"Page {node}: {score:.4f}")

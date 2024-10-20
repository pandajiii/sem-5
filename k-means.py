# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Create sample data (for clustering)
# Here, we generate a synthetic dataset with 3 clusters
X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# Apply KMeans Clustering
kmeans = KMeans(n_clusters=3)  # We want 3 clusters
kmeans.fit(X)

# Get the cluster centroids and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Plotting the results
plt.figure(figsize=(8,6))

# Plot the data points and assign colors based on their cluster labels
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)

# Plot the centroids of each cluster
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')

# Add labels and title
plt.title("K-Means Clustering", fontsize=15)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)

# Show the plot
plt.show()

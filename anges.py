import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

"""
You should use your own implementation of the AGNES algorithm using the Gower distance.
"""

data, y = make_blobs(n_samples=1000, centers=5, random_state=42)

agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg_clustering.fit_predict(data)

# Plot clusters
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolors='k')
plt.title("Agglomerative Clustering")
plt.show()

linked = linkage(data, method='ward')
plt.figure(figsize=(8, 5))
dendrogram(linked, orientation='top', distance_sort='ascending', show_leaf_counts=True)
plt.title("Dendrogram")
plt.show()

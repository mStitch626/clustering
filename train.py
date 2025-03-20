from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from utils.loads import evaluate_kmeans, kmeans

data, y = make_blobs(n_samples=1000, centers=5, random_state=42)
# plt.scatter(data[:, 0], data[:, 1], c=y, cmap='viridis', edgecolors='k')
# plt.title("Randomly Generated Data with 5 Clusters")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()

# evaluate_kmeans(data, start = 2,max_k=10)


k = 4
labels_custom, centroids_custom, inertia_custom = kmeans(data, k, 5)

# Run Default KMeans from sklearn
kmeans_sklearn = KMeans(n_clusters=k, random_state=42, n_init=10)
labels_sklearn = kmeans_sklearn.fit_predict(data)
centroids_sklearn = kmeans_sklearn.cluster_centers_
inertia_sklearn = kmeans_sklearn.inertia_

# Plot Comparison
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Custom KMeans Plot
axs[0].scatter(data[:, 0], data[:, 1], c=labels_custom, cmap='viridis', s=20, alpha=0.7)
axs[0].scatter(centroids_custom[:, 0], centroids_custom[:, 1], c='red', marker='X', s=200, label="Centroids")
axs[0].set_title(f"Custom K-Means (Inertia: {inertia_custom:.2f})")
axs[0].legend()

# Default Sklearn KMeans Plot
axs[1].scatter(data[:, 0], data[:, 1], c=labels_sklearn, cmap='viridis', s=20, alpha=0.7)
axs[1].scatter(centroids_sklearn[:, 0], centroids_sklearn[:, 1], c='red', marker='X', s=200, label="Centroids")
axs[1].set_title(f"Sklearn K-Means (Inertia: {inertia_sklearn:.2f})")
axs[1].legend()

plt.show()

print(f"Custom K-Means Inertia: {inertia_custom:.2f}")
print(f"Sklearn K-Means Inertia: {inertia_sklearn:.2f}")

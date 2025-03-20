import pandas as pd
from time import time

from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 


def load_csv_dataset(path, columns = None, take = None, **options):
    
    data = pd.read_csv(path, names=columns, **options)
    
    if take:
        pass
    
    return data

def fit_transform(scaler,data, **options):
    return pd.DataFrame(scaler.fit_transform(data), **options)


 
def kmeans(data, k, no_of_iterations=100):
    idx = np.random.choice(len(data), k, replace=False)
    centroids = data[idx, :]
    
    for _ in range(no_of_iterations): 
        distances = cdist(data, centroids, 'euclidean')
        points = np.array([np.argmin(i) for i in distances])
        
        new_centroids = []
        for i in range(k):
            cluster_points = data[points == i]
            if len(cluster_points) == 0:
                new_centroids.append(centroids[i])  # Keep the old centroid
            else:
                new_centroids.append(cluster_points.mean(axis=0))
        
        new_centroids = np.vstack(new_centroids)
        
        # Step 4: Check for convergence (if centroids do not change)
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids

    inertia = np.sum((data - centroids[points]) ** 2)

    return points, centroids, inertia

def visualize(data, kmeans, n_clusters):
    
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50, label="Data Points")
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')
    plt.title(f"K-Means Clustering with (n_clusters={n_clusters})")
    plt.legend()
    plt.show()


def evaluate_kmeans(data, start = 2,max_k=10):
    silhouette_scores = []
    intra_inertia = []
    inter_inertia = []
    global_centroid = np.mean(data, axis=0)  # CG_global

    for k in range(start, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        
        labels = kmeans.fit_predict(data)
        silhouette_scores.append(metrics.silhouette_score(data, labels))
    
        intra_inertia.append(kmeans.inertia_)

        centroids = kmeans.cluster_centers_
        inter_class_variance = np.sum([np.linalg.norm(c - global_centroid) ** 2 
                                       for c in centroids])
        inter_inertia.append(inter_class_variance)
        

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    k_values = range(start, max_k + 1)
    axs[0].plot(k_values, intra_inertia, marker='o', linestyle='-', color='b', label='Intra-Class Inertia (WCSS)')
    axs[0].set_xlabel("Number of Clusters (K)")
    axs[0].set_ylabel("Inertia")
    axs[0].set_title("Elbow Method")
    axs[0].legend()

    axs[1].plot(k_values, silhouette_scores, marker='o', linestyle='-', color='r', label='Silhouette Score')
    axs[1].set_xlabel("Number of Clusters (K)")
    axs[1].set_ylabel("Silhouette Score")
    axs[1].set_title("Silhouette Score Analysis")
    axs[1].legend()

    # axs[2].plot(k_values, inter_inertia, marker='s', linestyle='-', color='g', label='Inter-Class Inertia (BCV)')
    # axs[2].axvline(x=best_k, linestyle="--", color="red", label=f"Best K = {best_k}")
    # axs[2].set_xlabel("Number of Clusters (K)")
    # axs[2].set_ylabel("Inertia")
    # axs[2].set_title("Intra vs. Inter-Class Inertia")
    # axs[2].legend()
    
    plt.tight_layout()
    plt.show()


# contact = load_csv_dataset('./dataset/contact-lenses.csv', ['age', 'spectacle-prescrip', 'astigmatism', 'tear-prod-rate', 'contact-lenses'])

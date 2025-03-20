from sklearn.datasets import make_blobs
from .utils import evaluate_kmeans

data, _ = make_blobs(n_samples=1000, centers=4, random_state=42)


evaluate_kmeans(data, start = 2,max_k=10)
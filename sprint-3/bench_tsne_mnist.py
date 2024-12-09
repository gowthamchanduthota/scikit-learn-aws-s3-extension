"""
=============================
MNIST dataset T-SNE benchmark
=============================
"""

import argparse
import json
import os
import numpy as np
from time import time
from joblib import Memory
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array, shuffle
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

LOG_DIR = "mnist_tsne_output"
os.makedirs(LOG_DIR, exist_ok=True)

memory = Memory(os.path.join(LOG_DIR, "mnist_tsne_benchmark_data"), mmap_mode="r")


@memory.cache
def load_data(dtype=np.float32, order="C", shuffle_data=True, seed=0):
    """Load and preprocess the MNIST dataset."""
    print("Loading MNIST dataset...")
    data = fetch_openml("mnist_784", as_frame=False)
    X = check_array(data["data"], dtype=dtype, order=order) / 255  # Normalize features
    y = data["target"]

    if shuffle_data:
        X, y = shuffle(X, y, random_state=seed)

    return X, y


def compute_nn_accuracy(X, X_embedded):
    """Compute nearest-neighbor accuracy."""
    knn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
    neighbors_X = knn.fit(X).kneighbors(return_distance=False)
    neighbors_X_embedded = knn.fit(X_embedded).kneighbors(return_distance=False)
    return np.mean(neighbors_X == neighbors_X_embedded)


def fit_tsne(model, data):
    """Fit and transform data using t-SNE."""
    start_time = time()
    transformed = model.fit_transform(data)
    duration = time() - start_time
    return transformed, duration, model.n_iter_


def sanitize_filename(filename):
    """Sanitize filenames for safe saving."""
    return filename.replace("/", "-").replace(" ", "_")


def save_results(log_file, results):
    """Save results to a JSON file."""
    with open(log_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Benchmark for t-SNE")
    parser.add_argument("--order", default="C", help="Order of the input data")
    parser.add_argument("--perplexity", type=float, default=30, help="Perplexity for t-SNE")
    parser.add_argument("--all", action="store_true", help="Use the full MNIST dataset")
    parser.add_argument("--pca-components", type=int, default=50, help="Number of PCA components")
    args = parser.parse_args()

    print("Using {} threads".format(_openmp_effective_n_threads()))
    X, y = load_data(order=args.order)

    if args.pca_components > 0:
        print(f"Applying PCA with {args.pca_components} components...")
        t0 = time()
        X = PCA(n_components=args.pca_components).fit_transform(X)
        print(f"PCA preprocessing took {time() - t0:.3f}s")

    tsne = TSNE(
        n_components=2,
        init="pca",
        perplexity=args.perplexity,
        verbose=1,
        n_iter=1000,
    )

    dataset_sizes = [100, 500, 1000, 5000, 10000]
    if args.all:
        dataset_sizes.append(70000)

    results = []
    log_file = os.path.join(LOG_DIR, "mnist_tsne_results.json")

    for size in dataset_sizes:
        X_subset = X[:size]
        y_subset = y[:size]

        print(f"Processing {size} samples...")
        X_embedded, duration, n_iter = fit_tsne(tsne, X_subset)
        accuracy = compute_nn_accuracy(X_subset, X_embedded)

        print(
            f"t-SNE on {size} samples took {duration:.3f}s, "
            f"iterations: {n_iter}, nn accuracy: {accuracy:.3f}"
        )

        results.append({
            "method": "t-SNE",
            "samples": size,
            "duration": duration,
            "iterations": n_iter,
            "nn_accuracy": accuracy,
        })

        # Save embeddings and results
        sanitized_name = sanitize_filename("t-SNE")
        np.save(os.path.join(LOG_DIR, f"mnist_{sanitized_name}_{size}.npy"), X_embedded)
        save_results(log_file, results)


if __name__ == "__main__":
    main()

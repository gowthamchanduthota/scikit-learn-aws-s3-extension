import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = "mnist_tsne_output"

def load_data(embedding_path, labels_path):
    """Load embedding and label data from numpy files."""
    if not os.path.exists(embedding_path) or not os.path.exists(labels_path):
        raise FileNotFoundError("Embedding or label file not found.")
    
    X = np.load(embedding_path)
    y = np.load(labels_path)
    return X, y

def plot_embedding(X, y):
    """Plot the 2D embedding with labels."""
    unique_labels = np.unique(y)
    for label in unique_labels:
        mask = y == label
        plt.scatter(X[mask, 0], X[mask, 1], alpha=0.6, label=int(label), s=10)
    
    plt.legend(loc="best", markerscale=2, fontsize='small')
    plt.title("t-SNE Embedding of MNIST Data")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(alpha=0.3)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results for t-SNE")
    parser.add_argument(
        "--labels",
        type=str,
        default=os.path.join(LOG_DIR, "mnist_original_labels_10000.npy"),
        help="Path to 1D integer numpy array for labels",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default=os.path.join(LOG_DIR, "mnist_sklearn_TSNE_10000.npy"),
        help="Path to 2D float numpy array for embedded data",
    )
    args = parser.parse_args()

    # Load data
    X, y = load_data(args.embedding, args.labels)

    # Plot data
    plot_embedding(X, y)

if __name__ == "__main__":
    main()

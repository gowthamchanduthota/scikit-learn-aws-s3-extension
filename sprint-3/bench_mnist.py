<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
"""
=======================
MNIST dataset benchmark
=======================

Benchmark on the MNIST dataset. The dataset comprises 70,000 samples
and 784 features. Here, we consider the task of predicting
10 classes - digits from 0 to 9 from their raw images.

Classification performance:
===========================
Classifier               train-time   test-time   error-rate
------------------------------------------------------------
[..]
"""

import argparse
import os
from time import time

import numpy as np
from joblib import Memory

from sklearn.datasets import fetch_openml, get_data_home
=======
=======
>>>>>>> 23c541b (Create bench_mnist.py)
=======
>>>>>>> f7194ee (Create bench_mnist.py)
import argparse
import os
from time import time
from joblib import Memory
import numpy as np
from sklearn.datasets import fetch_openml
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> f7194ee (Create bench_mnist.py)
=======
=======
"""
=======================
MNIST dataset benchmark
=======================

Benchmark on the MNIST dataset. The dataset comprises 70,000 samples
and 784 features. Here, we consider the task of predicting
10 classes - digits from 0 to 9 from their raw images.

Classification performance:
===========================
Classifier               train-time   test-time   error-rate
------------------------------------------------------------
[..]
"""

import argparse
import os
from time import time

import numpy as np
from joblib import Memory

from sklearn.datasets import fetch_openml, get_data_home
>>>>>>> e6cf5d4 (Create bench_mnist.py)
>>>>>>> 23c541b (Create bench_mnist.py)
=======
>>>>>>> f7194ee (Create bench_mnist.py)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import zero_one_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_array
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

# Cache data for faster access
CACHE_DIR = os.path.join(get_data_home(), "mnist_benchmark_data")
=======
=======
>>>>>>> 23c541b (Create bench_mnist.py)
=======
>>>>>>> f7194ee (Create bench_mnist.py)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Cache data for faster access
CACHE_DIR = os.path.join(os.getenv('HOME'), ".cache", "mnist_benchmark_data")
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> f7194ee (Create bench_mnist.py)
=======
=======

# Cache data for faster access
CACHE_DIR = os.path.join(get_data_home(), "mnist_benchmark_data")
>>>>>>> e6cf5d4 (Create bench_mnist.py)
>>>>>>> 23c541b (Create bench_mnist.py)
=======
>>>>>>> f7194ee (Create bench_mnist.py)
memory = Memory(CACHE_DIR, mmap_mode="r")

@memory.cache
def load_data(dtype=np.float32, order="C"):
    """Load and normalize the MNIST dataset."""
    print("Loading and preparing dataset...")
    data = fetch_openml("mnist_784", as_frame=False)
    X = check_array(data["data"], dtype=dtype, order=order) / 255.0
    y = data["target"]
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    n_train = 60000
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]
=======
    return train_test_split(X, y, test_size=0.2, random_state=42)  # Split data into training and testing
>>>>>>> f7194ee (Create bench_mnist.py)
=======
    return train_test_split(X, y, test_size=0.2, random_state=42)  # Split data into training and testing
=======
    n_train = 60000
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]
>>>>>>> e6cf5d4 (Create bench_mnist.py)
>>>>>>> 23c541b (Create bench_mnist.py)
=======
    return train_test_split(X, y, test_size=0.2, random_state=42)  # Split data into training and testing
>>>>>>> f7194ee (Create bench_mnist.py)

# Define classifiers
ESTIMATORS = {
    "dummy": DummyClassifier(),
    "CART": DecisionTreeClassifier(),
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    "ExtraTrees": ExtraTreesClassifier(),
    "RandomForest": RandomForestClassifier(),
    "Nystroem-SVM": make_pipeline(
        Nystroem(gamma=0.015, n_components=1000), LinearSVC(C=100)
    ),
    "SampledRBF-SVM": make_pipeline(
        RBFSampler(gamma=0.015, n_components=1000), LinearSVC(C=100)
    ),
    "LogisticRegression-SAG": LogisticRegression(solver="sag", tol=1e-1, C=1e4),
    "LogisticRegression-SAGA": LogisticRegression(solver="saga", tol=1e-1, C=1e4),
=======
=======
>>>>>>> 23c541b (Create bench_mnist.py)
=======
>>>>>>> f7194ee (Create bench_mnist.py)
    "ExtraTrees": ExtraTreesClassifier(n_jobs=-1),  # Parallelize Extra Trees
    "RandomForest": RandomForestClassifier(n_jobs=-1),  # Parallelize Random Forest
    "Nystroem-SVM": make_pipeline(
        Nystroem(gamma=0.015, n_components=1000), LinearSVC(C=100, random_state=42)
    ),
    "SampledRBF-SVM": make_pipeline(
        RBFSampler(gamma=0.015, n_components=1000), LinearSVC(C=100, random_state=42)
    ),
    "LogisticRegression-SAG": LogisticRegression(solver="sag", tol=1e-1, C=1e4, random_state=42),
    "LogisticRegression-SAGA": LogisticRegression(solver="saga", tol=1e-1, C=1e4, random_state=42),
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> f7194ee (Create bench_mnist.py)
=======
=======
    "ExtraTrees": ExtraTreesClassifier(),
    "RandomForest": RandomForestClassifier(),
    "Nystroem-SVM": make_pipeline(
        Nystroem(gamma=0.015, n_components=1000), LinearSVC(C=100)
    ),
    "SampledRBF-SVM": make_pipeline(
        RBFSampler(gamma=0.015, n_components=1000), LinearSVC(C=100)
    ),
    "LogisticRegression-SAG": LogisticRegression(solver="sag", tol=1e-1, C=1e4),
    "LogisticRegression-SAGA": LogisticRegression(solver="saga", tol=1e-1, C=1e4),
>>>>>>> e6cf5d4 (Create bench_mnist.py)
>>>>>>> 23c541b (Create bench_mnist.py)
=======
>>>>>>> f7194ee (Create bench_mnist.py)
    "MultilayerPerceptron": MLPClassifier(
        hidden_layer_sizes=(100, 100),
        max_iter=400,
        alpha=1e-4,
        solver="sgd",
        learning_rate_init=0.2,
        momentum=0.9,
        tol=1e-4,
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        random_state=1,
=======
        random_state=42,
>>>>>>> f7194ee (Create bench_mnist.py)
=======
        random_state=42,
=======
        random_state=1,
>>>>>>> e6cf5d4 (Create bench_mnist.py)
>>>>>>> 23c541b (Create bench_mnist.py)
=======
        random_state=42,
>>>>>>> f7194ee (Create bench_mnist.py)
    ),
    "MLP-adam": MLPClassifier(
        hidden_layer_sizes=(100, 100),
        max_iter=400,
        alpha=1e-4,
        solver="adam",
        learning_rate_init=0.001,
        tol=1e-4,
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        random_state=1,
=======
        random_state=42,
>>>>>>> f7194ee (Create bench_mnist.py)
=======
        random_state=42,
=======
        random_state=1,
>>>>>>> e6cf5d4 (Create bench_mnist.py)
>>>>>>> 23c541b (Create bench_mnist.py)
=======
        random_state=42,
>>>>>>> f7194ee (Create bench_mnist.py)
    ),
}

def benchmark_classifiers(classifiers, X_train, X_test, y_train, y_test, n_jobs=1, random_seed=0):
    """Train, test, and evaluate classifiers."""
    results = {}
    for name in classifiers:
        print(f"Training {name}... ", end="")
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        estimator = ESTIMATORS[name]

        # Configure estimator
=======

        estimator = ESTIMATORS[name]
>>>>>>> f7194ee (Create bench_mnist.py)
=======

        estimator = ESTIMATORS[name]
=======
        estimator = ESTIMATORS[name]

        # Configure estimator
>>>>>>> e6cf5d4 (Create bench_mnist.py)
>>>>>>> 23c541b (Create bench_mnist.py)
=======

        estimator = ESTIMATORS[name]
>>>>>>> f7194ee (Create bench_mnist.py)
        params = estimator.get_params()
        if "random_state" in params:
            estimator.set_params(random_state=random_seed)
        if "n_jobs" in params:
            estimator.set_params(n_jobs=n_jobs)

        # Train
        start_time = time()
        estimator.fit(X_train, y_train)
        train_time = time() - start_time

        # Test
        start_time = time()
        y_pred = estimator.predict(X_test)
        test_time = time() - start_time

        # Compute error
        error = zero_one_loss(y_test, y_pred)
        results[name] = (train_time, test_time, error)
        print("done")
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> 23c541b (Create bench_mnist.py)
=======
>>>>>>> f7194ee (Create bench_mnist.py)

        # Print classification report for detailed metrics
        print(f"Classification report for {name}:\n{classification_report(y_test, y_pred)}\n")

<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> f7194ee (Create bench_mnist.py)
=======
=======
>>>>>>> e6cf5d4 (Create bench_mnist.py)
>>>>>>> 23c541b (Create bench_mnist.py)
=======
>>>>>>> f7194ee (Create bench_mnist.py)
    return results

def display_results(results):
    """Display benchmark results."""
    print("\nClassification performance:")
    print("===========================")
    print(f"{'Classifier':<24} {'Train Time':>10} {'Test Time':>10} {'Error Rate':>12}")
    print("-" * 60)
    for name, (train_time, test_time, error) in sorted(results.items(), key=lambda x: x[1][2]):
        print(f"{name:<23} {train_time:>10.2f}s {test_time:>10.2f}s {error:>12.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifiers",
        nargs="+",
        choices=ESTIMATORS.keys(),
        default=["ExtraTrees", "Nystroem-SVM"],
        help="List of classifiers to benchmark."
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel workers."
    )
    parser.add_argument(
        "--order",
        type=str,
        choices=["F", "C"],
        default="C",
        help="Data memory layout: Fortran ('F') or C ('C')."
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    print(__doc__)

    # Load data
    X_train, X_test, y_train, y_test = load_data(order=args.order)

    # Dataset statistics
    print(f"\nDataset statistics:\n{'='*20}")
    print(f"{'Number of features:':<25} {X_train.shape[1]}")
    print(f"{'Number of classes:':<25} {np.unique(y_train).size}")
    print(f"{'Number of train samples:':<25} {X_train.shape[0]} (size={X_train.nbytes // 1e6:.2f}MB)")
    print(f"{'Number of test samples:':<25} {X_test.shape[0]} (size={X_test.nbytes // 1e6:.2f}MB)")

    # Benchmark classifiers
    results = benchmark_classifiers(args.classifiers, X_train, X_test, y_train, y_test, args.n_jobs, args.random_seed)

    # Display results
    display_results(results)

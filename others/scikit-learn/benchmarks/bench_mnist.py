"""
=======================
MNIST dataset benchmark
=======================

<<<<<<< HEAD
Benchmark on the MNIST dataset. The dataset comprises 70,000 samples
and 784 features. Here, we consider the task of predicting
10 classes - digits from 0 to 9 from their raw images.

Classification performance:
===========================
Classifier               train-time   test-time   error-rate
------------------------------------------------------------
[..]
"""

=======
Benchmark on the MNIST dataset.  The dataset comprises 70,000 samples
and 784 features. Here, we consider the task of predicting
10 classes -  digits from 0 to 9 from their raw images. By contrast to the
covertype dataset, the feature space is homogeneous.

Example of output :
    [..]

    Classification performance:
    ===========================
    Classifier               train-time   test-time   error-rate
    ------------------------------------------------------------
    MLP_adam                     53.46s       0.11s       0.0224
    Nystroem-SVM                112.97s       0.92s       0.0228
    MultilayerPerceptron         24.33s       0.14s       0.0287
    ExtraTrees                   42.99s       0.57s       0.0294
    RandomForest                 42.70s       0.49s       0.0318
    SampledRBF-SVM              135.81s       0.56s       0.0486
    LinearRegression-SAG         16.67s       0.06s       0.0824
    CART                         20.69s       0.02s       0.1219
    dummy                         0.00s       0.01s       0.8973
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

>>>>>>> 60fa4732632c76d24e33583051c4348a682a2303
import argparse
import os
from time import time

import numpy as np
from joblib import Memory

from sklearn.datasets import fetch_openml, get_data_home
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
# Cache data for faster access
CACHE_DIR = os.path.join(get_data_home(), "mnist_benchmark_data")
memory = Memory(CACHE_DIR, mmap_mode="r")

@memory.cache
def load_data(dtype=np.float32, order="C"):
    """Load and normalize the MNIST dataset."""
    print("Loading and preparing dataset...")
    data = fetch_openml("mnist_784", as_frame=False)
    X = check_array(data["data"], dtype=dtype, order=order) / 255.0
    y = data["target"]
    n_train = 60000
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]

# Define classifiers
=======
# Memoize the data extraction and memory map the resulting
# train / test splits in readonly mode
memory = Memory(os.path.join(get_data_home(), "mnist_benchmark_data"), mmap_mode="r")


@memory.cache
def load_data(dtype=np.float32, order="F"):
    """Load the data, then cache and memmap the train/test split"""
    ######################################################################
    # Load dataset
    print("Loading dataset...")
    data = fetch_openml("mnist_784", as_frame=True)
    X = check_array(data["data"], dtype=dtype, order=order)
    y = data["target"]

    # Normalize features
    X = X / 255

    # Create train-test split (as [Joachims, 2006])
    print("Creating train-test split...")
    n_train = 60000
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    return X_train, X_test, y_train, y_test


>>>>>>> 60fa4732632c76d24e33583051c4348a682a2303
ESTIMATORS = {
    "dummy": DummyClassifier(),
    "CART": DecisionTreeClassifier(),
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
    "MultilayerPerceptron": MLPClassifier(
        hidden_layer_sizes=(100, 100),
        max_iter=400,
        alpha=1e-4,
        solver="sgd",
        learning_rate_init=0.2,
        momentum=0.9,
<<<<<<< HEAD
=======
        verbose=1,
>>>>>>> 60fa4732632c76d24e33583051c4348a682a2303
        tol=1e-4,
        random_state=1,
    ),
    "MLP-adam": MLPClassifier(
        hidden_layer_sizes=(100, 100),
        max_iter=400,
        alpha=1e-4,
        solver="adam",
        learning_rate_init=0.001,
<<<<<<< HEAD
=======
        verbose=1,
>>>>>>> 60fa4732632c76d24e33583051c4348a682a2303
        tol=1e-4,
        random_state=1,
    ),
}

<<<<<<< HEAD
def benchmark_classifiers(classifiers, X_train, X_test, y_train, y_test, n_jobs=1, random_seed=0):
    """Train, test, and evaluate classifiers."""
    results = {}
    for name in classifiers:
        print(f"Training {name}... ", end="")
        estimator = ESTIMATORS[name]

        # Configure estimator
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
    return results

def display_results(results):
    """Display benchmark results."""
    print("\nClassification performance:")
    print("===========================")
    print(f"{'Classifier':<24} {'Train Time':>10} {'Test Time':>10} {'Error Rate':>12}")
    print("-" * 60)
    for name, (train_time, test_time, error) in sorted(results.items(), key=lambda x: x[1][2]):
        print(f"{name:<23} {train_time:>10.2f}s {test_time:>10.2f}s {error:>12.4f}")
=======
>>>>>>> 60fa4732632c76d24e33583051c4348a682a2303

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifiers",
        nargs="+",
<<<<<<< HEAD
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
=======
        choices=ESTIMATORS,
        type=str,
        default=["ExtraTrees", "Nystroem-SVM"],
        help="list of classifiers to benchmark.",
    )
    parser.add_argument(
        "--n-jobs",
        nargs="?",
        default=1,
        type=int,
        help=(
            "Number of concurrently running workers for "
            "models that support parallelism."
        ),
    )
    parser.add_argument(
        "--order",
        nargs="?",
        default="C",
        type=str,
        choices=["F", "C"],
        help="Allow to choose between fortran and C ordered data",
    )
    parser.add_argument(
        "--random-seed",
        nargs="?",
        default=0,
        type=int,
        help="Common seed used by random number generator.",
    )
    args = vars(parser.parse_args())

    print(__doc__)

    X_train, X_test, y_train, y_test = load_data(order=args["order"])

    print("")
    print("Dataset statistics:")
    print("===================")
    print("%s %d" % ("number of features:".ljust(25), X_train.shape[1]))
    print("%s %d" % ("number of classes:".ljust(25), np.unique(y_train).size))
    print("%s %s" % ("data type:".ljust(25), X_train.dtype))
    print(
        "%s %d (size=%dMB)"
        % (
            "number of train samples:".ljust(25),
            X_train.shape[0],
            int(X_train.nbytes / 1e6),
        )
    )
    print(
        "%s %d (size=%dMB)"
        % (
            "number of test samples:".ljust(25),
            X_test.shape[0],
            int(X_test.nbytes / 1e6),
        )
    )

    print()
    print("Training Classifiers")
    print("====================")
    error, train_time, test_time = {}, {}, {}
    for name in sorted(args["classifiers"]):
        print("Training %s ... " % name, end="")
        estimator = ESTIMATORS[name]
        estimator_params = estimator.get_params()

        estimator.set_params(
            **{
                p: args["random_seed"]
                for p in estimator_params
                if p.endswith("random_state")
            }
        )

        if "n_jobs" in estimator_params:
            estimator.set_params(n_jobs=args["n_jobs"])

        time_start = time()
        estimator.fit(X_train, y_train)
        train_time[name] = time() - time_start

        time_start = time()
        y_pred = estimator.predict(X_test)
        test_time[name] = time() - time_start

        error[name] = zero_one_loss(y_test, y_pred)

        print("done")

    print()
    print("Classification performance:")
    print("===========================")
    print(
        "{0: <24} {1: >10} {2: >11} {3: >12}".format(
            "Classifier  ", "train-time", "test-time", "error-rate"
        )
    )
    print("-" * 60)
    for name in sorted(args["classifiers"], key=error.get):
        print(
            "{0: <23} {1: >10.2f}s {2: >10.2f}s {3: >12.4f}".format(
                name, train_time[name], test_time[name], error[name]
            )
        )

    print()
>>>>>>> 60fa4732632c76d24e33583051c4348a682a2303

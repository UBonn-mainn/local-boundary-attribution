from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def generate_linearly_separable_data(
    n_samples_per_class: int = 200,
    n_features: int = 2,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a simple linearly separable dataset with two classes.

    Parameters
    ----------
    n_samples_per_class : int
        Number of samples per class.
    n_features : int
        Number of features. Default is 2 so the data can be visualized.
    random_state : int or None
        If given, use as RNG seed for reproducibility.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (2 * n_samples_per_class, n_features).
    y : np.ndarray
        Label vector of shape (2 * n_samples_per_class,) with values {0, 1}.
    """
    if random_state is not None:
        np.random.seed(random_state)
    else:
        # Seed based on current time (like your original script)
        from time import time
        np.random.seed(int(time()))

    # Generate Class 0
    mean_class0 = np.random.uniform(-5, 0, size=n_features)
    cov_class0 = np.eye(n_features) * np.random.uniform(0.5, 1.5)
    X_class0 = np.random.multivariate_normal(mean_class0, cov_class0, n_samples_per_class)
    y_class0 = np.zeros(n_samples_per_class)

    # Generate Class 1
    mean_class1 = np.random.uniform(2, 7, size=n_features)
    cov_class1 = np.eye(n_features) * np.random.uniform(0.5, 1.5)
    X_class1 = np.random.multivariate_normal(mean_class1, cov_class1, n_samples_per_class)
    y_class1 = np.ones(n_samples_per_class)

    # Combine
    X = np.vstack((X_class0, X_class1))
    y = np.hstack((y_class0, y_class1))

    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    return X, y


def save_dataset_to_csv(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: str | Path = ".",
    prefix: str = "linearly_separable_data",
    timestamp: Optional[str] = None,
) -> Path:
    """
    Save a dataset (X, y) to a CSV file.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Label vector of shape (n_samples,).
    out_dir : str or Path
        Directory where the CSV file will be saved.
    prefix : str
        Prefix for the filename.
    timestamp : str or None
        If None, current timestamp (YYYYMMDD_HHMMSS) is used.

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = out_dir / f"{prefix}_{timestamp}.csv"

    data_to_save = np.column_stack((X, y))
    header = ",".join([f"feature{i+1}" for i in range(X.shape[1])]) + ",class"

    np.savetxt(filename, data_to_save, delimiter=",", header=header, comments="")

    return filename


def plot_2d_dataset(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: str | Path = ".",
    prefix: str = "linearly_separable_plot",
    timestamp: Optional[str] = None,
    show: bool = True,
) -> Optional[Path]:
    """
    Visualize a 2D dataset and optionally save the plot.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, 2).
    y : np.ndarray
        Label vector of shape (n_samples,).
    out_dir : str or Path
        Directory where the PNG file will be saved.
    prefix : str
        Prefix for the plot filename.
    timestamp : str or None
        If None, current timestamp (YYYYMMDD_HHMMSS) is used.
    show : bool
        Whether to display the plot using plt.show().

    Returns
    -------
    Path or None
        Path to the saved PNG file if saved; otherwise None.
    """
    if X.shape[1] != 2:
        raise ValueError("plot_2d_dataset requires X to have exactly 2 features.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plot_path = out_dir / f"{prefix}_{timestamp}.png"

    plt.figure(figsize=(10, 6))
    plt.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        c="blue",
        label="Class 0",
        alpha=0.6,
        edgecolors="k",
    )
    plt.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        c="red",
        label="Class 1",
        alpha=0.6,
        edgecolors="k",
    )
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Linearly Separable 2D Data")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close()

    return plot_path


def load_dataset_from_csv(csv_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a dataset from a CSV file.
    Assumes the last column is the target variable.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file.

    Returns
    -------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Label vector.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found at {csv_path}")

    # Load CSV using numpy, skipping header
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    # Split into X and y. Assumes last column is class.
    X = data[:, :-1].astype(np.float32)
    y = data[:, -1].astype(np.float32)

    return X, y


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate linear separable dataset")
    parser.add_argument("--n_samples", type=int, default=200, help="Number of samples per class")
    parser.add_argument("--n_features", type=int, default=2, help="Number of features")
    parser.add_argument("--out_dir", type=str, default=".", help="Output directory")
    parser.add_argument("--no_plot", action="store_true", help="Disable plotting")
    
    args = parser.parse_args()
    
    print("Generating data...")
    X, y = generate_linearly_separable_data(
        n_samples_per_class=args.n_samples, 
        n_features=args.n_features
    )
    
    print("Saving data...")
    csv_path = save_dataset_to_csv(X, y, out_dir=args.out_dir)
    print(f"Data saved to: {csv_path}")
    
    if not args.no_plot and args.n_features == 2:
        print("Plotting data...")
        plot_path = plot_2d_dataset(X, y, out_dir=args.out_dir, show=True)
        if plot_path:
            print(f"Plot saved to: {plot_path}")


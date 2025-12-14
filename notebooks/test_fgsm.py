import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from types import SimpleNamespace

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

import torch
import torch.nn.functional as F

# --- IMPORT YOUR UTILS / BOUNDARY SEARCH ---
from boundary_search.fgsm import fgsm_boundary_search
from utils.models.train_linear_model import train
from utils.data.load_model import load_model
from utils.data.dataset_utils import load_dataset_from_csv

# 1. LOAD DATA FROM CSV ---------------------------------------------------------
# Re-using utils.dataset_utils.load_dataset_from_csv directly

# 2. FIND BOUNDARY POINTS FOR SOME SAMPLES -------------------------------------

def compute_boundary_points(model, X, n_samples=20, device="cpu"):
    """
    Pick n_samples random points from X and compute FGSM boundary points.
    Returns:
        samples:  (n_samples, 2)
        boundaries: (n_samples, 2)
        success_mask: (n_samples,) bool
    """
    model.eval()
    idx = np.random.choice(len(X), size=n_samples, replace=False)
    X_sel = X[idx]

    samples = []
    boundaries = []
    success_mask = []

    for x_np in X_sel:
        x_t = torch.from_numpy(x_np).to(device)
        bnd, success = fgsm_boundary_search(
            model,
            x_t,
            step_size=0.1,
            max_iters=50,
            clamp=None,
            refine_steps=15,
        )
        samples.append(x_np)
        boundaries.append(bnd.cpu().numpy())
        success_mask.append(success)

    return np.array(samples), np.array(boundaries), np.array(success_mask, dtype=bool)


# 3. PLOTTING FUNCTIONS ---------------------------------------------------------

def plot_data_with_boundaries(X, y, samples, boundaries, success_mask, output_path=None):
    """
    Plot:
    - all data points
    - a subset of 'samples'
    - their boundary points
    - a line between sample and boundary
    """
    plt.figure(figsize=(8, 6))

    # full dataset
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="blue", label="Class 0", alpha=0.3)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="red", label="Class 1", alpha=0.3)

    # only successful boundary points
    samples_s = samples[success_mask]
    boundaries_s = boundaries[success_mask]

    # connect each sample to its boundary
    for s, b in zip(samples_s, boundaries_s):
        plt.plot(
            [s[0], b[0]],
            [s[1], b[1]],
            "k--",
            linewidth=1,
            alpha=0.7,
        )

    # highlight the samples and boundary points
    plt.scatter(
        samples_s[:, 0],
        samples_s[:, 1],
        c="green",
        marker="o",
        edgecolors="k",
        label="Selected samples",
    )
    plt.scatter(
        boundaries_s[:, 0],
        boundaries_s[:, 1],
        c="yellow",
        marker="X",
        edgecolors="k",
        s=80,
        label="Boundary points (FGSM)",
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Samples and FGSM-Based Boundary Points (Linear Model)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to {output_path}")

    plt.show()


def plot_decision_regions(model, X, y, device="cpu", resolution=200, output_path=None,
                          samples=None, boundaries=None, success_mask=None):
    """
    Plot model's decision regions over the 2D space.
    Optionally overlays samples and their boundary points.
    """
    model.eval()

    x1_min, x1_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    x2_min, x2_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, resolution),
        np.linspace(x2_min, x2_max, resolution),
    )

    grid = np.c_[xx1.ravel(), xx2.ravel()].astype(np.float32)
    with torch.no_grad():
        grid_t = torch.from_numpy(grid).to(device)
        logits = model(grid_t)
        probs = F.softmax(logits, dim=1)
        # Use probability of class 1 for gradient visualization
        preds = probs[:, 1].cpu().numpy()

    zz = preds.reshape(xx1.shape)

    plt.figure(figsize=(8, 6))
    # Plot probability gradient
    contour = plt.contourf(xx1, xx2, zz, alpha=0.8, levels=np.linspace(0, 1, 21), cmap="RdBu_r")
    plt.colorbar(contour, label="Probability of Class 1")
    
    # Add decision boundary line
    plt.contour(xx1, xx2, zz, levels=[0.5], colors="black", linewidths=2, linestyles="--")
    
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="blue", edgecolors="k", label="Class 0", alpha=0.6)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="red", edgecolors="k", label="Class 1", alpha=0.6)

    # Overlay boundary paths if provided
    if samples is not None and boundaries is not None and success_mask is not None:
        samples_s = samples[success_mask]
        boundaries_s = boundaries[success_mask]

        # connect each sample to its boundary
        for s, b in zip(samples_s, boundaries_s):
            plt.plot([s[0], b[0]], [s[1], b[1]], "k--", linewidth=1, alpha=0.7)

        # highlight the samples and boundary points
        plt.scatter(samples_s[:, 0], samples_s[:, 1], c="green", marker="o", edgecolors="k", label="Selected samples")
        plt.scatter(boundaries_s[:, 0], boundaries_s[:, 1], c="yellow", marker="X", edgecolors="k", s=80, label="Boundary points")


    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Model Decision Gradient & Boundary Paths")
    # De-duplicate legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to {output_path}")

    plt.show()


# 4. MAIN -----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test FGSM boundary search on Linear Model")
    parser.add_argument("--n_samples", type=int, default=30, help="Number of samples to test")
    parser.add_argument("--data_csv", type=str, default=None, help="Path to data CSV (optional)")
    args_script = parser.parse_args()

    # 1) set the path to the CSV
    project_root = Path(__file__).resolve().parent.parent
    
    if args_script.data_csv:
        csv_path = args_script.data_csv
    else:
        # Auto-detect default
        csv_path = "/Users/nguyennhatmai/Documents/study/UBonn/WiSe2526/LabDMAI/local-boundary-attribution/data/synthetic/linearly_separable_data_20251207_012853.csv"
        data_dir = project_root / "data" / "synthetic"
        if data_dir.exists():
            csv_files = list(data_dir.glob("*.csv"))
            if csv_files:
                csv_path = str(csv_files[0])
                print(f"Auto-detected data file: {csv_path}")

    # 2) train model using the centralized script
    save_path = project_root / "models" / "checkpoints" / "linear_model_fgsm_test.pth"
    
    print(f"Training on {csv_path}...")
    
    # Create args object for training
    train_args = SimpleNamespace(
        data_path=str(csv_path),
        save_path=str(save_path),
        epochs=100,
        batch_size=32,
        lr=0.01,
        val_split=0.2,
        seed=42
    )
    
    train(train_args)
    
    # 3) Load the trained model
    print(f"Loading model from {save_path}...")
    model = load_model(str(save_path), input_dim=2, model_type="linear")
    
    # 4) Load data for plotting
    X, y = load_dataset_from_csv(csv_path)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    device = "cpu" 
    
    # 5) compute boundary points
    print(f"Computing boundary points for {args_script.n_samples} samples...")
    samples, boundaries, success_mask = compute_boundary_points(model, X, n_samples=args_script.n_samples, device=device)
    print(f"Boundary search success for {success_mask.sum()}/{len(success_mask)} samples")

    # 6) plot figures
    figures_dir = project_root / "figures"
    decision_plot_path = figures_dir / "test_fgsm_decision_regions.png"
    boundary_plot_path = figures_dir / "test_fgsm_boundary_points.png"
    
    # Plot decision regions with gradient AND paths
    plot_decision_regions(model, X, y, device=device, output_path=str(decision_plot_path),
                          samples=samples, boundaries=boundaries, success_mask=success_mask)
    
    plot_data_with_boundaries(X, y, samples, boundaries, success_mask, output_path=str(boundary_plot_path))

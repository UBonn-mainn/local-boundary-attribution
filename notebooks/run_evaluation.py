import argparse
import csv
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from boundary_search.fgsm import fgsm_boundary_search
from evaluation.decision_boundary_metrics import (
    neutrality_metric,
    proximity_metric,
    stability_metric,
    flip_correctness_metric,
)


# ----------------------------
# Data & Model helpers
# ----------------------------

def load_csv_dataset(csv_path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assumes columns: feature1,feature2,...,class
    """
    csv_path = Path(csv_path)
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    X = data[:, :-1].astype(np.float32)
    y = data[:, -1].astype(np.int64)
    return X, y


class SimpleMLP(nn.Module):
    """
    Small MLP for 2D synthetic data.
    Replace with your real model if needed.
    """

    def __init__(self, in_dim=2, hidden_dim=32, n_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train_or_load_model(
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-2,
    checkpoint_path = None,
) -> nn.Module:
    """
    If checkpoint_path exists, load model from there.
    Otherwise, train a SimpleMLP on (X, y) and optionally save.
    """
    model = SimpleMLP(in_dim=X.shape[1], hidden_dim=32, n_classes=2).to(device)

    if checkpoint_path is not None and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        print(f"[INFO] Loaded model weights from {checkpoint_path}")
        return model

    print("[INFO] Training a new model...")
    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:03d} | loss = {total_loss / len(dataset):.4f}")

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[INFO] Saved model to {checkpoint_path}")

    return model


# ----------------------------
# Plot helpers
# ----------------------------

def plot_histogram(values: np.ndarray, out_path: Path, title: str, xlabel: str):
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved histogram to {out_path}")


def plot_samples_and_boundaries(
    X: np.ndarray,
    y: np.ndarray,
    X_samples: np.ndarray,
    X_boundaries: np.ndarray,
    out_path: Path,
):
    """
    Only for 2D data.
    Plots:
    - dataset in the background,
    - selected samples,
    - boundary points,
    - dashed lines from sample -> boundary.
    """
    if X.shape[1] != 2:
        print("[WARN] plot_samples_and_boundaries: X is not 2D, skipping scatter plot.")
        return

    plt.figure(figsize=(7, 6))

    # background dataset
    plt.scatter(
        X[y == 0, 0],
        X[y == 0, 1],
        c="blue",
        alpha=0.3,
        label="Class 0",
        edgecolors="none",
    )
    plt.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        c="red",
        alpha=0.3,
        label="Class 1",
        edgecolors="none",
    )

    # lines and points
    for s, b in zip(X_samples, X_boundaries):
        plt.plot([s[0], b[0]], [s[1], b[1]], "k--", linewidth=1, alpha=0.7)

    plt.scatter(
        X_samples[:, 0],
        X_samples[:, 1],
        c="green",
        edgecolors="k",
        label="Samples",
    )
    plt.scatter(
        X_boundaries[:, 0],
        X_boundaries[:, 1],
        c="yellow",
        edgecolors="k",
        marker="X",
        s=80,
        label="Boundary points",
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Samples and Decision-Boundary Points")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved scatter plot to {out_path}")


# ----------------------------
# Main evaluation
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate decision boundary search metrics.")
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to CSV dataset (feature1,...,class).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/evaluation",
        help="Output directory for CSVs and logs.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional model checkpoint path (if exists, will be loaded).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum number of samples to evaluate.",
    )
    parser.add_argument(
        "--stability-runs",
        type=int,
        default=5,
        help="Number of runs per sample for stability metric.",
    )
    parser.add_argument(
        "--stability-samples",
        type=int,
        default=20,
        help="Number of samples (subset) used for stability evaluation.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # --- Load data ---
    X, y = load_csv_dataset(csv_path)
    print(f"[INFO] Loaded dataset from {csv_path} with shape X={X.shape}, y={y.shape}")

    max_samples = min(args.max_samples, len(X))
    X_eval = X[:max_samples]
    y_eval = y[:max_samples]

    # --- Load or train model ---
    checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else None
    model = train_or_load_model(X, y, device=device, checkpoint_path=checkpoint_path)
    model.eval()

    # --- Boundary search function (FGSM here; swap with IBS/CMA-ES if needed) ---
    search_fn = fgsm_boundary_search

    # --- Per-sample metrics ---
    per_sample_rows = []
    neutrality_vals = []
    proximity_vals = []
    flip_correct_vals = []
    success_flags = []
    boundary_points = []  # store boundary points (or None) for plotting later

    for i in range(max_samples):
        x_np = X_eval[i]
        label = int(y_eval[i])
        x_t = torch.from_numpy(x_np).to(device)

        boundary_x, success = search_fn(model, x_t)

        if not success:
            neutrality = float("nan")
            proximity = float("nan")
            flip_ok = False
            boundary_points.append(None)
        else:
            # 1) Neutrality
            neutrality = neutrality_metric(
                model=model,
                x_boundary=boundary_x,
                x_original=x_t,
                cls_mode="original",  # or "max"
            )

            # 2) Proximity
            proximity = proximity_metric(
                x_original=x_t,
                x_boundary=boundary_x,
                p=2.0,
            )

            # 4) Flip correctness
            flip_info = flip_correctness_metric(
                model=model,
                x_original=x_t,
                x_boundary=boundary_x,
                eps=1e-2,
            )
            flip_ok = bool(flip_info["flip_correct"])

            boundary_points.append(boundary_x.detach().cpu().numpy())

        neutrality_vals.append(neutrality)
        proximity_vals.append(proximity)
        flip_correct_vals.append(flip_ok)
        success_flags.append(success)

        per_sample_rows.append(
            {
                "index": i,
                "label": label,
                "success": int(success),
                "neutrality": neutrality,
                "proximity_l2": proximity,
                "flip_correct": int(flip_ok),
            }
        )

    # --- Stability on a subset ---
    stability_indices = np.random.choice(
        max_samples,
        size=min(args.stability_samples, max_samples),
        replace=False,
    )
    stability_results = []

    for idx in stability_indices:
        x_np = X_eval[idx]
        x_t = torch.from_numpy(x_np).to(device)

        stab = stability_metric(
            model=model,
            x=x_t,
            search_fn=search_fn,
            n_runs=args.stability_runs,
            device=device,
        )
        stability_results.append(stab)

    # Aggregate stability
    if len(stability_results) > 0:
        mean_success_rate = float(np.mean([s.success_rate for s in stability_results]))
        mean_pairwise = float(np.nanmean([s.mean_pairwise_distance for s in stability_results]))
        mean_std_to_mean = float(np.nanmean([s.std_to_mean for s in stability_results]))
    else:
        mean_success_rate = float("nan")
        mean_pairwise = float("nan")
        mean_std_to_mean = float("nan")

    # --- Aggregate dataset-level metrics ---
    success_arr = np.array(success_flags, dtype=bool)
    flip_arr = np.array(flip_correct_vals, dtype=bool)
    neutrality_arr = np.array(neutrality_vals, dtype=float)
    proximity_arr = np.array(proximity_vals, dtype=float)

    mask = success_arr & ~np.isnan(neutrality_arr) & ~np.isnan(proximity_arr)

    if mask.any():
        mean_neutrality = float(np.mean(neutrality_arr[mask]))
        median_neutrality = float(np.median(neutrality_arr[mask]))
        mean_proximity = float(np.mean(proximity_arr[mask]))
        median_proximity = float(np.median(proximity_arr[mask]))
        flip_rate = float(flip_arr[mask].mean())
    else:
        mean_neutrality = median_neutrality = float("nan")
        mean_proximity = median_proximity = float("nan")
        flip_rate = float("nan")

    success_rate = float(success_arr.mean())

    # --- Save per-sample CSV ---
    per_sample_csv = out_dir / "boundary_metrics_per_sample.csv"
    with per_sample_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "label", "success", "neutrality", "proximity_l2", "flip_correct"],
        )
        writer.writeheader()
        for row in per_sample_rows:
            writer.writerow(row)

    print(f"[INFO] Saved per-sample metrics to {per_sample_csv}")

    # --- Save summary CSV ---
    summary_csv = out_dir / "boundary_metrics_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["success_rate", success_rate])
        writer.writerow(["mean_neutrality", mean_neutrality])
        writer.writerow(["median_neutrality", median_neutrality])
        writer.writerow(["mean_proximity_l2", mean_proximity])
        writer.writerow(["median_proximity_l2", median_proximity])
        writer.writerow(["flip_correct_rate", flip_rate])
        writer.writerow(["stability_mean_success_rate", mean_success_rate])
        writer.writerow(["stability_mean_pairwise_dist", mean_pairwise])
        writer.writerow(["stability_mean_std_to_mean", mean_std_to_mean])

    print(f"[INFO] Saved summary metrics to {summary_csv}")

    # --- PLOTS: histograms & scatter ---------------------

    if mask.any():
        # Histograms (only successful, finite values)
        neut_vals = neutrality_arr[mask]
        prox_vals = proximity_arr[mask]

        plot_histogram(
            neut_vals,
            out_dir / "neutrality_histogram.png",
            title="Neutrality Distribution",
            xlabel="| P_c(x') - 0.5 |",
        )

        plot_histogram(
            prox_vals,
            out_dir / "proximity_histogram.png",
            title="L2 Proximity Distribution",
            xlabel="|| x - x' ||_2",
        )

        # Scatter plot for 2D data only
        if X.shape[1] == 2:
            X_succ = X_eval[mask]
            # select matching boundary points
            bnd_succ = np.stack(
                [bp for bp, m in zip(boundary_points, mask) if (m and bp is not None)],
                axis=0,
            )
            plot_samples_and_boundaries(
                X=X,
                y=y,
                X_samples=X_succ,
                X_boundaries=bnd_succ,
                out_path=out_dir / "samples_vs_boundaries.png",
            )
        else:
            print("[WARN] Data is not 2D; skipping scatter plot of samples vs boundary points.")
    else:
        print("[WARN] No successful boundary points to plot.")

    # --- Print LaTeX table snippet ---
    # print("\n=== LaTeX Table Snippet ===")
    # print(r"\begin{tabular}{l r}")
    # print(r"\toprule")
    # print(r"Metric & Value \\")
    # print(r"\midrule")
    # print(f"Success rate & {success_rate:.3f} \\\\")
    # print(f"Mean neutrality & {mean_neutrality:.4f} \\\\")
    # print(f"Median neutrality & {median_neutrality:.4f} \\\\")
    # print(f"Mean $L_2$ proximity & {mean_proximity:.4f} \\\\")
    # print(f"Median $L_2$ proximity & {median_proximity:.4f} \\\\")
    # print(f"Flip correctness rate & {flip_rate:.3f} \\\\")
    # print(f"Stability success rate & {mean_success_rate:.3f} \\\\")
    # print(f"Stability mean pairwise dist. & {mean_pairwise:.4f} \\\\")
    # print(f"Stability std-to-mean & {mean_std_to_mean:.4f} \\\\")
    # print(r"\bottomrule")
    # print(r"\end{tabular}")
    # print("=== End LaTeX Snippet ===")


if __name__ == "__main__":
    main()

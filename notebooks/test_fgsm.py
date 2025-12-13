import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# --- IMPORT YOUR UTILS / BOUNDARY SEARCH ---
from boundary_search.fgsm import fgsm_boundary_search


# 1. LOAD DATA FROM CSV ---------------------------------------------------------

def load_csv_dataset(csv_path):
    """
    Read CSV created by generate_linearly_separable_data + save_dataset_to_csv.
    Assumes columns: feature1,feature2,class
    """
    csv_path = Path(csv_path)
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    X = data[:, :-1].astype(np.float32)
    y = data[:, -1].astype(np.int64)
    return X, y


# 2. SIMPLE MODEL & TRAINING ----------------------------------------------------

class SimpleMLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=16, n_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train_model(X, y, epochs=200, batch_size=32, lr=1e-2, device="cpu"):
    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleMLP(in_dim=X.shape[1], hidden_dim=16, n_classes=2).to(device)
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

    return model


# 3. FIND BOUNDARY POINTS FOR SOME SAMPLES -------------------------------------

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


# 4. PLOTTING FUNCTIONS ---------------------------------------------------------

def plot_data_with_boundaries(X, y, samples, boundaries, success_mask):
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
    plt.title("Samples and FGSM-Based Boundary Points")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_decision_regions(model, X, y, device="cpu", resolution=200):
    """
    Plot model's decision regions over the 2D space.
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
        preds = probs.argmax(dim=1).cpu().numpy()

    zz = preds.reshape(xx1.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx1, xx2, zz, alpha=0.3, levels=[-0.5, 0.5, 1.5], colors=["blue", "red"])
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="blue", edgecolors="k", label="Class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="red", edgecolors="k", label="Class 1")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Model Decision Regions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


# 5. MAIN -----------------------------------------------------------------------

if __name__ == "__main__":
    # 1) set the path to the CSV you generated earlier
    csv_path = "/Users/nguyennhatmai/Documents/study/UBonn/WiSe2526/LabDMAI/local-boundary-attribution/data/synthetic/linearly_separable_data_20251207_012853.csv"

    # 2) load
    X, y = load_csv_dataset(csv_path)
    print("Loaded data:", X.shape, y.shape)

    device = "cpu"

    # 3) train model
    model = train_model(X, y, epochs=200, batch_size=32, lr=1e-2, device=device)

    # 4) compute boundary points
    samples, boundaries, success_mask = compute_boundary_points(model, X, n_samples=30, device=device)
    print(f"Boundary search success for {success_mask.sum()}/{len(success_mask)} samples")

    # 5) plot figures
    plot_decision_regions(model, X, y, device=device)
    plt.show()

    plot_data_with_boundaries(X, y, samples, boundaries, success_mask)
    plt.show()

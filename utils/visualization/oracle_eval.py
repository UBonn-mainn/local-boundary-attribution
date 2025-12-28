from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def _predict_label_np(model: torch.nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    """Predict labels for X (N,d) -> (N,) as numpy."""
    x_t = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(x_t)
    return logits.argmax(dim=1).detach().cpu().numpy()


def plot_2d_boundary_comparison(
    model: torch.nn.Module,
    x: np.ndarray,
    b_fgsm: Optional[np.ndarray],
    b_gs: Optional[np.ndarray],
    save_path: str | Path,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    padding: float = 0.25,
    grid_res: int = 250,
    device: Optional[torch.device] = None,
    title: Optional[str] = None,
) -> None:
    """
    Visualize model decision regions + x + FGSM/GS boundary points (2D only).
    Saves a single PNG/PDF.

    - No explicit colors: uses matplotlib defaults.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    device = device or next(model.parameters()).device

    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.shape[0] != 2:
        raise ValueError("plot_2d_boundary_comparison requires 2D inputs.")

    points = [x]
    if b_fgsm is not None:
        points.append(np.asarray(b_fgsm, dtype=np.float32).reshape(-1))
    if b_gs is not None:
        points.append(np.asarray(b_gs, dtype=np.float32).reshape(-1))
    points = np.stack(points, axis=0)

    if X_train is not None and X_train.shape[1] == 2:
        xmin, ymin = X_train.min(axis=0)
        xmax, ymax = X_train.max(axis=0)
    else:
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)

    dx = (xmax - xmin) if xmax > xmin else 1.0
    dy = (ymax - ymin) if ymax > ymin else 1.0
    xmin -= padding * dx
    xmax += padding * dx
    ymin -= padding * dy
    ymax += padding * dy

    xs = np.linspace(xmin, xmax, grid_res)
    ys = np.linspace(ymin, ymax, grid_res)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

    preds = _predict_label_np(model, grid, device=device)
    Z = preds.reshape(grid_res, grid_res)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, levels=len(np.unique(preds)) + 1, alpha=0.25)
    ax.contour(xx, yy, Z, levels=len(np.unique(preds)) + 1, linewidths=1.0)

    if X_train is not None and y_train is not None and X_train.shape[1] == 2:
        ax.scatter(X_train[:, 0], X_train[:, 1], s=10, alpha=0.35)

    ax.scatter([x[0]], [x[1]], marker="o", s=90, label="x")

    if b_fgsm is not None:
        b_fgsm = np.asarray(b_fgsm, dtype=np.float32).reshape(-1)
        ax.scatter([b_fgsm[0]], [b_fgsm[1]], marker="x", s=120, label="FGSM boundary")
        ax.plot([x[0], b_fgsm[0]], [x[1], b_fgsm[1]], linewidth=1.5)

    if b_gs is not None:
        b_gs = np.asarray(b_gs, dtype=np.float32).reshape(-1)
        ax.scatter([b_gs[0]], [b_gs[1]], marker="+", s=160, label="GS oracle boundary")
        ax.plot([x[0], b_gs[0]], [x[1], b_gs[1]], linewidth=1.5)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title or "Decision regions + boundary points")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_distance_summaries(
    df,
    save_dir: str | Path,
    prefix: str = "fgsm_vs_gs",
) -> None:
    """
    df must contain columns:
      - dist_x_to_fgsm_boundary
      - dist_x_to_gs_boundary
      - dist_fgsm_to_gs_boundary
    Saves histograms as PNG.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cols = [
        "dist_x_to_fgsm_boundary",
        "dist_x_to_gs_boundary",
        "dist_fgsm_to_gs_boundary",
    ]
    for c in cols:
        if c not in df.columns:
            continue
        fig, ax = plt.subplots()
        ax.hist(df[c].dropna().values, bins=30)
        ax.set_title(c)
        ax.set_xlabel("distance")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(save_dir / f"{prefix}_{c}.png", dpi=200)
        plt.close(fig)

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def _predict_label_np(model: torch.nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    x_t = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(x_t)
    return logits.argmax(dim=1).detach().cpu().numpy()


def _try_get_linear_params(model: torch.nn.Module) -> Optional[Tuple[np.ndarray, float]]:
    """
    Try to extract (w, b) for a 2-class linear boundary w^T x + b = 0.

    Supports:
      - torch.nn.Linear directly
      - wrappers that contain a `.linear` module
    Returns None if not possible.
    """
    linear = None

    if isinstance(model, torch.nn.Linear):
        linear = model
    elif hasattr(model, "linear") and isinstance(getattr(model, "linear"), torch.nn.Linear):
        linear = getattr(model, "linear")

    if linear is None:
        return None

    W = linear.weight.detach().cpu().numpy()
    b = linear.bias.detach().cpu().numpy() if linear.bias is not None else np.zeros((W.shape[0],), dtype=np.float32)

    # If multi-class, boundary depends on which classes. For visualization, we pick class0 vs class1 if available.
    if W.shape[0] >= 2:
        w = (W[1] - W[0]).astype(np.float32)
        bb = float(b[1] - b[0])
        return w, bb

    # Binary with single logit: boundary at logit=0
    if W.shape[0] == 1:
        w = W[0].astype(np.float32)
        bb = float(b[0])
        return w, bb

    return None


def _fit_linear_surrogate_from_grid(
    xx: np.ndarray,
    yy: np.ndarray,
    Z: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Fit a linear separator w^T x + b = 0 from grid labels Z using least squares on {-1,+1}.
    This is a *surrogate* when the model is not linear.
    """
    Xg = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)  # (N,2)
    y = Z.ravel().astype(np.int64)

    # Map majority/minority into -1/+1 (works for 2-class; for multi-class, this becomes crude)
    classes = np.unique(y)
    if len(classes) < 2:
        # degenerate
        return np.array([1.0, 0.0], dtype=np.float32), 0.0

    # choose two most frequent classes
    counts = [(c, int((y == c).sum())) for c in classes]
    counts.sort(key=lambda t: t[1], reverse=True)
    c_pos, c_neg = counts[0][0], counts[1][0]

    y_pm = np.where(y == c_pos, 1.0, -1.0).astype(np.float32)

    # Solve [X, 1] theta ≈ y_pm
    A = np.concatenate([Xg, np.ones((Xg.shape[0], 1), dtype=np.float32)], axis=1)  # (N,3)
    theta, *_ = np.linalg.lstsq(A, y_pm, rcond=None)
    w = theta[:2].astype(np.float32)
    b = float(theta[2])
    return w, b


def _plot_linear_boundary(ax, w: np.ndarray, b: float, xlim: Tuple[float, float], ylim: Tuple[float, float], label: str):
    """
    Plot line w1*x + w2*y + b = 0 without specifying a color.
    """
    w1, w2 = float(w[0]), float(w[1])
    xmin, xmax = xlim
    ymin, ymax = ylim

    # If w2 != 0, solve for y = -(w1*x + b)/w2
    if abs(w2) > 1e-8:
        xs = np.array([xmin, xmax], dtype=np.float32)
        ys = -(w1 * xs + b) / w2
        ax.plot(xs, ys, linestyle="--", linewidth=2.0, label=label)
    else:
        # vertical line x = -b / w1
        if abs(w1) > 1e-8:
            x0 = -b / w1
            ax.plot([x0, x0], [ymin, ymax], linestyle="--", linewidth=2.0, label=label)


def plot_2d_boundary_comparison( # show gradient color
    model: torch.nn.Module,
    x: np.ndarray,
    b_crawler: Optional[np.ndarray],
    b_gs: Optional[np.ndarray],
    save_path: str | Path,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    padding: float = 0.25,
    grid_res: int = 250,
    device: Optional[torch.device] = None,
    title: Optional[str] = None,
    # circles
    gs_radius: Optional[float] = None,
    show_gs_sphere: bool = True,
    crawler_circle: bool = True,
    crawler_circle_radius: Optional[float] = None,
    # boundary
    show_linear_boundary: bool = True,
) -> None:
    """
    Visualize:
      - model decision regions + contour
      - point x
      - Crawler boundary point and segment x->b_crawler
      - GS boundary point and segment x->b_gs
      - GS sphere (circle) centered at x with radius=gs_radius (optional)
      - Crawler circle centered at x with radius=||x-b_crawler|| (optional, ensures b_crawler lies on circle)
      - linear decision boundary line (exact if available, else surrogate) (optional)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    device = device or next(model.parameters()).device

    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.shape[0] != 2:
        raise ValueError("plot_2d_boundary_comparison requires 2D inputs.")

    # Convert boundary points
    bf = None if b_crawler is None else np.asarray(b_crawler, dtype=np.float32).reshape(-1)
    bg = None if b_gs is None else np.asarray(b_gs, dtype=np.float32).reshape(-1)

    # Crawler circle radius (default: ||x-b_crawler||)
    r_crawler = None
    if crawler_circle:
        if crawler_circle_radius is not None:
            r_crawler = float(crawler_circle_radius)
        elif bf is not None:
            r_crawler = float(np.linalg.norm(bf - x))

    # Determine plot bounds (include circles so they don't get clipped)
    pts = [x]
    if bf is not None:
        pts.append(bf)
    if bg is not None:
        pts.append(bg)
    pts = np.stack(pts, axis=0)

    if X_train is not None and X_train.shape[1] == 2:
        xmin, ymin = X_train.min(axis=0)
        xmax, ymax = X_train.max(axis=0)
    else:
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)

    # Expand bounds to include circles
    extra_r = 0.0
    if show_gs_sphere and gs_radius is not None and np.isfinite(gs_radius) and gs_radius > 0:
        extra_r = max(extra_r, float(gs_radius))
    if crawler_circle and r_crawler is not None and np.isfinite(r_crawler) and r_crawler > 0:
        extra_r = max(extra_r, float(r_crawler))

    dx = (xmax - xmin) if xmax > xmin else 1.0
    dy = (ymax - ymin) if ymax > ymin else 1.0

    # base padding plus circle padding
    xmin = min(xmin - padding * dx, x[0] - extra_r * (1.0 + padding))
    xmax = max(xmax + padding * dx, x[0] + extra_r * (1.0 + padding))
    ymin = min(ymin - padding * dy, x[1] - extra_r * (1.0 + padding))
    ymax = max(ymax + padding * dy, x[1] + extra_r * (1.0 + padding))

    # Decision grid
    xs = np.linspace(xmin, xmax, grid_res)
    ys = np.linspace(ymin, ymax, grid_res)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

    preds = _predict_label_np(model, grid, device=device)
    Z = preds.reshape(grid_res, grid_res)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, levels=len(np.unique(preds)) + 1, alpha=0.25)
    ax.contour(xx, yy, Z, levels=len(np.unique(preds)) + 1, linewidths=1.0)

    # Optional training scatter
    if X_train is not None and y_train is not None and X_train.shape[1] == 2:
        ax.scatter(X_train[:, 0], X_train[:, 1], s=10, alpha=0.35)

    # GS sphere (circle)
    if show_gs_sphere and gs_radius is not None and np.isfinite(gs_radius) and gs_radius > 0:
        ax.add_patch(
            plt.Circle((x[0], x[1]), float(gs_radius), fill=False, linewidth=2.0, label="GS sphere")
        )

    # Crawler circle centered at x and passing through b_crawler
    if crawler_circle and r_crawler is not None and np.isfinite(r_crawler) and r_crawler > 0:
        ax.add_patch(
            plt.Circle((x[0], x[1]), float(r_crawler), fill=False, linewidth=2.0, label="Crawler circle (|x-b_crawler|)")
        )

    # Linear decision boundary (exact if possible else surrogate)
    if show_linear_boundary:
        params = _try_get_linear_params(model)
        if params is not None:
            w, b = params
            _plot_linear_boundary(ax, w, b, xlim=(xmin, xmax), ylim=(ymin, ymax), label="Linear boundary (model)")
        else:
            w, b = _fit_linear_surrogate_from_grid(xx, yy, Z)
            _plot_linear_boundary(ax, w, b, xlim=(xmin, xmax), ylim=(ymin, ymax), label="Linear boundary (surrogate)")

    # Points + segments
    ax.scatter([x[0]], [x[1]], marker="o", s=90, label="x")

    if bf is not None:
        ax.scatter([bf[0]], [bf[1]], marker="x", s=120, label="Crawler boundary")
        ax.plot([x[0], bf[0]], [x[1], bf[1]], linewidth=1.5)

    if bg is not None:
        ax.scatter([bg[0]], [bg[1]], marker="+", s=160, label="GS oracle boundary")
        ax.plot([x[0], bg[0]], [x[1], bg[1]], linewidth=1.5)

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
    prefix: str = "crawler_vs_gs",
) -> None:
    """
    df must contain columns:
      - dist_x_to_crawler_boundary
      - dist_x_to_gs_boundary
      - dist_crawler_to_gs_boundary
    Saves histograms as PNG.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cols = [
        "dist_x_to_crawler_boundary",
        "dist_x_to_gs_boundary",
        "dist_crawler_to_gs_boundary",
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

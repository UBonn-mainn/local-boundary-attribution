from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import torch


@torch.no_grad()
def _predict_labels(model: torch.nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(Xt)
    return logits.argmax(dim=1).detach().cpu().numpy()


def count_classes_in_ring(
    *,
    x: np.ndarray,
    b_fgsm: np.ndarray,
    r_gs: float,
    X_ref: np.ndarray,
    y_ref: Optional[np.ndarray] = None,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Count points per class in the annulus between the GS and FGSM radii around x.

    Ring is defined over reference points X_ref (e.g., your dataset points).
    You can count by:
      - ground truth labels (if y_ref is provided)
      - model predicted labels (if model is provided)

    Returns counts + sizes:
      {
        "ring_n": int,
        "ring_r_min": float,
        "ring_r_max": float,
        "ring_count_y0": int, "ring_count_y1": int, ... (if y_ref)
        "ring_count_pred0": int, "ring_count_pred1": int, ... (if model)
      }
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    b_fgsm = np.asarray(b_fgsm, dtype=np.float32).reshape(-1)
    X_ref = np.asarray(X_ref, dtype=np.float32)

    r_fgsm = float(np.linalg.norm(b_fgsm - x))
    r_gs = float(r_gs)

    if not np.isfinite(r_fgsm) or not np.isfinite(r_gs) or r_fgsm <= eps or r_gs <= eps:
        return {
            "ring_n": 0,
            "ring_r_min": float("nan"),
            "ring_r_max": float("nan"),
        }

    r_min, r_max = (r_fgsm, r_gs) if r_fgsm < r_gs else (r_gs, r_fgsm)

    d = np.linalg.norm(X_ref - x[None, :], axis=1)
    mask = (d >= r_min) & (d <= r_max)
    idx = np.where(mask)[0]

    out: Dict[str, float] = {
        "ring_n": int(idx.size),
        "ring_r_min": float(r_min),
        "ring_r_max": float(r_max),
    }

    if idx.size == 0:
        return out

    # ---- counts by ground-truth labels ----
    if y_ref is not None:
        y_ref = np.asarray(y_ref)
        y_ring = y_ref[idx]
        classes = np.unique(y_ref)
        for c in classes:
            out[f"ring_count_y{int(c)}"] = int(np.sum(y_ring == c))

    # ---- counts by model-predicted labels ----
    if model is not None:
        if device is None:
            device = next(model.parameters()).device
        preds = _predict_labels(model, X_ref[idx], device=device)
        classes = np.unique(preds)  # classes present in ring
        for c in classes:
            out[f"ring_count_pred{int(c)}"] = int(np.sum(preds == c))

    return out

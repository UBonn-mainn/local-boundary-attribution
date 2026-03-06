import numpy as np
import torch
from typing import Dict


@torch.no_grad()
def _predict_model(model, Z: np.ndarray, device) -> np.ndarray:
    Zt = torch.tensor(Z, dtype=torch.float32, device=device)
    logits = model(Zt)
    return logits.argmax(dim=1).cpu().numpy()


def _sample_points_in_sphere(x: np.ndarray, r: float, N: int, rng: np.random.Generator):
    """
    Uniform sampling in an L2 ball.
    """
    d = x.shape[0]
    v = rng.normal(size=(N, d))
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    u = rng.random(N) ** (1.0 / d)
    return x[None, :] + r * u[:, None] * v


def compute_left_right_area_disagreement(
    model: torch.nn.Module,
    x: np.ndarray,
    b_fgsm: np.ndarray,
    b_gs: np.ndarray,
    r_gs: float,
    N: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Monte Carlo estimator of left/right area disagreement.

    Returns:
      {
        "area_left": float,
        "area_right": float,
        "area_mean": float
      }
    """
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)

    x = np.asarray(x, dtype=np.float32)
    b_fgsm = np.asarray(b_fgsm, dtype=np.float32)
    b_gs = np.asarray(b_gs, dtype=np.float32)

    # --- directions ---
    u = b_gs - x
    u = u / (np.linalg.norm(u) + 1e-12)          # GS direction
    n = b_fgsm - x                                # FGSM normal

    # --- sample ---
    Z = _sample_points_in_sphere(x, r_gs, N, rng)

    # --- split left / right ---
    proj = (Z - x[None, :]) @ u
    mask_right = proj >= 0
    mask_left = proj < 0

    # --- true model labels ---
    fz = _predict_model(model, Z, device=device)

    # --- FGSM-induced labels ---
    g_scores = (Z - b_fgsm[None, :]) @ n
    gz = (g_scores >= 0).astype(fz.dtype)

    # Align FGSM labels with model labels at x
    fx = _predict_model(model, x[None, :], device=device)[0]
    if gz[mask_right][0] != fx:
        gz = 1 - gz

    # --- disagreements ---
    def _area(mask):
        if mask.sum() == 0:
            return 0.0
        return float(np.mean(fz[mask] != gz[mask]))

    area_left = _area(mask_left)
    area_right = _area(mask_right)

    return {
        "area_left": area_left,
        "area_right": area_right,
        "area_mean": 0.5 * (area_left + area_right),
    }

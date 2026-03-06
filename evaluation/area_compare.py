import numpy as np
import torch
from typing import Dict


@torch.no_grad()
def _predict_model(model: torch.nn.Module, Z: np.ndarray, device) -> np.ndarray:
    Zt = torch.tensor(Z, dtype=torch.float32, device=device)
    logits = model(Zt)
    return logits.argmax(dim=1).detach().cpu().numpy()


def _sample_points_in_ball(x: np.ndarray, r: float, N: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform sampling in an L2 ball."""
    d = x.shape[0]
    v = rng.normal(size=(N, d)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    u = rng.random(N).astype(np.float32) ** (1.0 / d)
    return x[None, :] + (r * u)[:, None] * v


def _induced_labels(z: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Binary labels from hyperplane through b with normal n."""
    scores = (z - b[None, :]) @ n
    return (scores >= 0).astype(np.int64)


def compute_area_fgsm_vs_gs(
    model: torch.nn.Module,
    x: np.ndarray,
    b_fgsm: np.ndarray,
    b_gs: np.ndarray,
    r_gs: float,
    N: int = 4000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Computes:
      - area_fgsm: fraction of ball classified as f(x) by FGSM-induced hyperplane
      - area_gs:   fraction of ball classified as f(x) by GS-induced hyperplane
      - area_disagreement: fraction where FGSM-induced and GS-induced differ

    Ball is centered at x with radius r_gs (from GS).
    """
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)

    x = np.asarray(x, dtype=np.float32)
    b_fgsm = np.asarray(b_fgsm, dtype=np.float32)
    b_gs = np.asarray(b_gs, dtype=np.float32)

    if not np.isfinite(r_gs) or r_gs <= 0:
        return {"area_fgsm": np.nan, "area_gs": np.nan, "area_disagreement": np.nan}

    # Sample region Omega (GS ball)
    Z = _sample_points_in_ball(x, float(r_gs), N, rng)

    # Reference label at x (true model)
    fx = int(_predict_model(model, x[None, :], device=device)[0])

    # FGSM hyperplane
    n_f = b_fgsm - x
    if np.linalg.norm(n_f) < 1e-12:
        return {"area_fgsm": np.nan, "area_gs": np.nan, "area_disagreement": np.nan}

    gf = _induced_labels(Z, b_fgsm, n_f)

    # Align gf so that gf(x) == f(x)
    gf_x = _induced_labels(x[None, :], b_fgsm, n_f)[0]
    if gf_x != fx:
        gf = 1 - gf

    # GS hyperplane
    n_g = b_gs - x
    if np.linalg.norm(n_g) < 1e-12:
        return {"area_fgsm": float(np.mean(gf == fx)), "area_gs": np.nan, "area_disagreement": np.nan}

    gg = _induced_labels(Z, b_gs, n_g)

    gg_x = _induced_labels(x[None, :], b_gs, n_g)[0]
    if gg_x != fx:
        gg = 1 - gg

    area_fgsm = float(np.mean(gf == fx))
    area_gs = float(np.mean(gg == fx))
    area_disagreement = float(np.mean(gf != gg))

    return {
        "area_fgsm": area_fgsm,
        "area_gs": area_gs,
        "area_disagreement": area_disagreement,
    }

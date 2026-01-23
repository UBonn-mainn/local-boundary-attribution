import numpy as np
import torch
import math


@torch.no_grad()
def _predict_label_np(model: torch.nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    return model(Xt).argmax(dim=1).detach().cpu().numpy()


def _sample_uniform_d_ball(x: np.ndarray, r: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Uniform sampling in d-dimensional L2 ball centered at x with radius r.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    d = x.shape[0]

    # random directions
    u = rng.normal(size=(n, d))
    u_norm = np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
    v = u / u_norm

    # radii with correct distribution
    t = rng.random(n)
    rho = r * (t ** (1.0 / d))

    return x[None, :] + v * rho[:, None]


def _log_volume_unit_d_ball(d: int) -> float:
    """
    log Vol(B_d(0,1)) = (d/2) log(pi) - log Gamma(d/2 + 1)
    Uses lgamma for stability.
    """
    return 0.5 * d * math.log(math.pi) - math.lgamma(0.5 * d + 1.0)


def decision_boundary_sphere_volume(
    model: torch.nn.Module,
    x: np.ndarray,
    r: float,
    device: torch.device,
    n_samples: int = 20000,
    seed: int = 0,
    return_absolute_volume: bool = True,
) -> dict:
    """
    Computes:
      - red_frac: fraction of d-ball classified differently than class(model(x))
      - red_vol:  absolute volume (optional), using analytic d-ball volume

    Works for any dimension d.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    d = x.shape[0]
    r = float(r)

    if not np.isfinite(r) or r <= 1e-12:
        return {"red_frac": np.nan, "red_vol": np.nan}

    rng = np.random.default_rng(seed)

    cx = int(_predict_label_np(model, x[None, :].astype(np.float32), device=device)[0])
    Z = _sample_uniform_d_ball(x, r, n_samples, rng).astype(np.float32)
    lab = _predict_label_np(model, Z, device=device)

    red_frac = float(np.mean(lab != cx))

    if not return_absolute_volume:
        return {"red_frac": red_frac, "red_vol": np.nan}

    # Vol(B_d(x,r)) = Vol(B_d(0,1)) * r^d
    logV = _log_volume_unit_d_ball(d) + d * math.log(r)
    ball_vol = math.exp(logV)
    red_vol = ball_vol * red_frac

    return {"red_frac": red_frac, "red_vol": float(red_vol)}

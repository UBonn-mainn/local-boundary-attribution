import numpy as np
import torch
import math


@torch.no_grad()
def _predict_label_np(model: torch.nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)

    if X.ndim == 1:
        X = X[None, :]

    if X.ndim == 2:
        d = X.shape[1]
        side = int(np.sqrt(d))
        if side * side != d:
            raise ValueError(f"Cannot reshape feature dimension {d} into square image.")
        Xt = torch.tensor(X, dtype=torch.float32, device=device).view(-1, 1, side, side)

    elif X.ndim == 4:
        Xt = torch.tensor(X, dtype=torch.float32, device=device)

    else:
        raise ValueError(f"Unsupported input shape: {X.shape}")

    return model(Xt).argmax(dim=1).detach().cpu().numpy()


def _sample_uniform_d_ball(x: np.ndarray, r: float, n: int, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    d = x.shape[0]

    u = rng.normal(size=(n, d))
    u_norm = np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
    v = u / u_norm

    t = rng.random(n)
    rho = r * (t ** (1.0 / d))

    return x[None, :] + v * rho[:, None]


def _log_volume_unit_d_ball(d: int) -> float:
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
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    d = x.shape[0]
    r = float(r)

    if not np.isfinite(r) or r <= 1e-12:
        return {"red_frac": np.nan, "red_vol": np.nan}

    rng = np.random.default_rng(seed)

    cx = int(_predict_label_np(model, x[None, :].astype(np.float32), device=device)[0])
    Z = _sample_uniform_d_ball(x, r, n_samples, rng).astype(np.float32)
    Z = np.clip(Z, 0.0, 1.0)
    lab = _predict_label_np(model, Z, device=device)

    red_frac = float(np.mean(lab != cx))

    if not return_absolute_volume:
        return {"red_frac": red_frac, "red_vol": np.nan}

    logV = _log_volume_unit_d_ball(d) + d * math.log(r)
    ball_vol = math.exp(logV)
    red_vol = ball_vol * red_frac

    return {"red_frac": red_frac, "red_vol": float(red_vol)}
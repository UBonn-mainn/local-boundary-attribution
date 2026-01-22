import numpy as np
from typing import Tuple


def angle_at_x_degrees(x: np.ndarray, b_fgsm: np.ndarray, b_gs: np.ndarray, eps: float = 1e-12) -> float:
    """
    Angle (in degrees) between vectors (b_fgsm - x) and (b_gs - x), measured at x.
    Returns np.nan if any vector has ~zero length.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    bf = np.asarray(b_fgsm, dtype=np.float32).reshape(-1)
    bg = np.asarray(b_gs, dtype=np.float32).reshape(-1)

    v1 = bf - x
    v2 = bg - x

    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < eps or n2 < eps:
        return float("nan")

    cos = float(np.dot(v1, v2) / (n1 * n2))
    cos = float(np.clip(cos, -1.0, 1.0))  # numerical safety
    theta = float(np.degrees(np.arccos(cos)))
    return theta

from typing import Dict, List, Any

import numpy as np


def summarize_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    rows: list of dicts with scalar metrics.
    Returns dict: metric -> {mean, std, median, q10, q90}
    """
    if not rows:
        return {}

    # collect scalar keys
    keys = sorted({k for r in rows for k in r.keys() if isinstance(r[k], (int, float, np.number))})
    out: Dict[str, Dict[str, float]] = {}

    for k in keys:
        vals = np.array([float(r[k]) for r in rows if k in r], dtype=np.float64)
        if vals.size == 0:
            continue
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "median": float(np.median(vals)),
            "q10": float(np.quantile(vals, 0.10)),
            "q90": float(np.quantile(vals, 0.90)),
        }
    return out
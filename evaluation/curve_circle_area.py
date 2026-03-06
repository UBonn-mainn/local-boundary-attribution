from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import torch

@torch.no_grad()
def _predict_label_np(model, X: np.ndarray, device: torch.device) -> np.ndarray:
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    return model(xt).argmax(dim=1).detach().cpu().numpy()


def _sample_uniform_disk(center: np.ndarray, radius: float, n: int, rng: np.random.Generator):
    # Uniform in disk: r = R*sqrt(u), theta = 2pi v
    u = rng.random(n)
    v = rng.random(n)
    rr = radius * np.sqrt(u)
    th = 2.0 * np.pi * v
    pts = np.stack([rr * np.cos(th), rr * np.sin(th)], axis=1)
    return pts + center.reshape(1, 2)


def cal_curve_circle_area(
        model: torch.nn.Module,
        x: np.ndarray,
        radius: float,
        device: torch.device,
        n_samples: int = 20000,
        seed: int = 0,
):
    """
    Area inside FGSM disk that is classified differently than class(model(x)).
    Returns (area_red, frac_red).
    """
    x = np.asarray(x, dtype=np.float32).reshape(2, )
    r = float(radius)
    if not np.isfinite(r) or r <= 1e-12:
        return {"curve_circle_area": np.nan, "curve_circle_frac": np.nan}

    rng = np.random.default_rng(seed)
    cx = int(_predict_label_np(model, x.reshape(1, 2), device=device)[0])

    pts = _sample_uniform_disk(x, r, n_samples, rng)
    lab = _predict_label_np(model, pts, device=device)

    curve_circle_frac = float(np.mean(lab != cx))
    curve_circle_area = float(np.pi * r * r * curve_circle_frac)
    return {"curve_circle_area": curve_circle_area, "curve_circle_frac": curve_circle_frac}


# ----------------------- Helpers: ordering boundary points into a curve -----------------------

def order_boundary_points_by_pca(points: np.ndarray) -> np.ndarray:
    """
    Order boundary points into a polyline by projecting onto the first PCA axis.
    Works well if the boundary is roughly line-like (common for linearly separable 2D).

    points: (N,2)
    returns: (N,2) ordered
    """
    P = np.asarray(points, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("order_boundary_points_by_pca expects (N,2) points")

    mean = P.mean(axis=0, keepdims=True)
    X = P - mean
    # PCA: top eigenvector of covariance
    C = (X.T @ X) / max(len(P) - 1, 1)
    vals, vecs = np.linalg.eigh(C)
    v = vecs[:, np.argmax(vals)]  # (2,)
    t = X @ v  # (N,)
    idx = np.argsort(t)
    return P[idx]


# ----------------------- Geometry: segment-circle intersections -----------------------

@dataclass
class Intersection:
    seg_idx: int  # segment index i for segment P[i] -> P[i+1]
    t: float  # param along segment in [0,1]
    point: np.ndarray  # (2,)


def _segment_circle_intersections(p0: np.ndarray, p1: np.ndarray, c: np.ndarray, r: float) -> List[
    Tuple[float, np.ndarray]]:
    """
    Return intersections between segment p0->p1 and circle centered at c with radius r.
    Returns list of (t, point) with t in [0,1].
    """
    d = p1 - p0
    f = p0 - c

    a = float(np.dot(d, d))
    b = 2.0 * float(np.dot(f, d))
    cc = float(np.dot(f, f) - r * r)

    disc = b * b - 4.0 * a * cc
    if disc < 0.0 or a < 1e-15:
        return []

    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    out = []
    if 0.0 <= t1 <= 1.0:
        out.append((float(t1), p0 + t1 * d))
    if 0.0 <= t2 <= 1.0 and (disc > 1e-12):  # avoid duplicate when tangent
        out.append((float(t2), p0 + t2 * d))
    return out


def find_polyline_circle_intersections(poly: np.ndarray, c: np.ndarray, r: float) -> List[Intersection]:
    """
    poly: (M,2) ordered polyline points (open polyline)
    Returns all intersections with segments poly[i]->poly[i+1].
    """
    P = np.asarray(poly, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64).reshape(2, )
    r = float(r)

    inters: List[Intersection] = []
    for i in range(len(P) - 1):
        hits = _segment_circle_intersections(P[i], P[i + 1], c, r)
        for t, pt in hits:
            inters.append(Intersection(seg_idx=i, t=t, point=pt))
    return inters


# ----------------------- Area: boundary segment + circle arc -----------------------

def _wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _shoelace_area(poly: np.ndarray) -> float:
    """Area of polygon given by vertices poly (N,2), assumed ordered."""
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _circle_arc_points(c: np.ndarray, r: float, p_start: np.ndarray, p_end: np.ndarray, n: int = 200,
                       use_shorter_arc: bool = True) -> np.ndarray:
    """
    Points along circle arc from p_start to p_end (both on circle), around center c.
    """
    c = np.asarray(c, dtype=np.float64).reshape(2, )
    p_start = np.asarray(p_start, dtype=np.float64).reshape(2, )
    p_end = np.asarray(p_end, dtype=np.float64).reshape(2, )
    r = float(r)

    a0 = np.arctan2(p_start[1] - c[1], p_start[0] - c[0])
    a1 = np.arctan2(p_end[1] - c[1], p_end[0] - c[0])

    d = _wrap_to_pi(a1 - a0)  # shortest signed difference
    if use_shorter_arc:
        a_end = a0 + d
    else:
        # longer arc is the other way around
        if d > 0:
            d_long = d - 2.0 * np.pi
        else:
            d_long = d + 2.0 * np.pi
        a_end = a0 + d_long

    ang = np.linspace(a0, a_end, n)
    arc = np.stack([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)], axis=1)
    return arc


def curve_circle_enclosed_area(
        curve: np.ndarray,
        x: np.ndarray,
        b_fgsm_x: np.ndarray,
        arc_resolution: int = 200,
        use_shorter_arc: bool = True,
) -> float:
    """
    Compute the area enclosed by:
      - the portion of the boundary curve inside the circle centered at x with radius ||b_fgsm_x-x||
      - the circle arc connecting the two intersection points

    Returns np.nan if fewer than 2 intersections exist.

    Assumes curve is an ordered open polyline (M,2).
    """
    curve = np.asarray(curve, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).reshape(2, )
    b_fgsm_x = np.asarray(b_fgsm_x, dtype=np.float64).reshape(2, )

    r = float(np.linalg.norm(b_fgsm_x - x))
    if not np.isfinite(r) or r <= 1e-12:
        return float("nan")

    inters = find_polyline_circle_intersections(curve, x, r)
    if len(inters) < 2:
        return float("nan")

    # Choose a pair of intersections that yields the "shortest" curve segment between them (in index-distance).
    # This is typically the local piece that cuts through the circle.
    inters_sorted = sorted(inters, key=lambda I: (I.seg_idx, I.t))

    best = None
    best_cost = 1e18
    for i in range(len(inters_sorted)):
        for j in range(i + 1, len(inters_sorted)):
            I0, I1 = inters_sorted[i], inters_sorted[j]
            # cost: number of segments between them
            cost = (I1.seg_idx - I0.seg_idx) + abs(I1.t - I0.t)
            if cost < best_cost:
                best_cost = cost
                best = (I0, I1)

    I0, I1 = best  # intersection endpoints on circle

    # Build the boundary segment polyline between I0 and I1
    seg_points = []
    seg_points.append(I0.point)
    # add interior vertices between segments
    for k in range(I0.seg_idx + 1, I1.seg_idx + 1):
        seg_points.append(curve[k])
    seg_points.append(I1.point)
    seg_points = np.asarray(seg_points, dtype=np.float64)

    # Build circle arc from I1 back to I0 to close the loop (or vice versa—either works with abs area)
    arc = _circle_arc_points(x, r, p_start=I1.point, p_end=I0.point, n=arc_resolution, use_shorter_arc=use_shorter_arc)

    # Closed polygon: boundary segment + arc
    poly = np.vstack([seg_points, arc])

    return float(_shoelace_area(poly))
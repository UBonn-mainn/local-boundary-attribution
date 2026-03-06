from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, List

import torch
import torch.nn.functional as F


# Type alias: boundary search function that works on a single sample x
BoundarySearchFn = Callable[
    [torch.nn.Module, torch.Tensor],
    Tuple[torch.Tensor, bool]
]


# ----------------------------
# 1. Neutrality
# ----------------------------

@torch.no_grad()
def neutrality_metric(
    model: torch.nn.Module,
    x_boundary: torch.Tensor,
    x_original: Optional[torch.Tensor] = None,
    cls_mode: str = "original",
) -> float:
    """
    Neutrality of a boundary point x_boundary.

    Neutrality = | P_c(x_boundary) - 0.5 |,  where c is either:
      - the original predicted class at x_original (cls_mode="original")
      - the argmax class at x_boundary (cls_mode="max")

    Lower is better (0 means perfectly neutral).

    Args
    ----
    model : torch.nn.Module
        Classifier, outputs logits.
    x_boundary : torch.Tensor
        Boundary point, no batch dimension.
    x_original : torch.Tensor or None
        Original input sample. Required if cls_mode="original".
    cls_mode : {"original", "max"}
        How to choose the class c for the neutrality measure.

    Returns
    -------
    float
        Neutrality score (>= 0).
    """
    model.eval()

    if x_boundary.dim() == 1:
        xb = x_boundary.unsqueeze(0)
    else:
        xb = x_boundary

    probs_b = F.softmax(model(xb), dim=1)[0]

    if cls_mode == "original":
        if x_original is None:
            raise ValueError("x_original must be provided when cls_mode='original'.")
        if x_original.dim() == 1:
            xo = x_original.unsqueeze(0)
        else:
            xo = x_original
        probs_x = F.softmax(model(xo), dim=1)[0]
        cls = probs_x.argmax().item()
    elif cls_mode == "max":
        cls = probs_b.argmax().item()
    else:
        raise ValueError(f"Unknown cls_mode='{cls_mode}'.")

    p_cls = probs_b[cls].item()
    return abs(p_cls - 0.5)


# ----------------------------
# 2. Proximity (distance to original)
# ----------------------------

def proximity_metric(
    x_original: torch.Tensor,
    x_boundary: torch.Tensor,
    p: float = 2.0,
) -> float:
    """
    Proximity between original point and boundary point.

    Typically L2 distance: ||x - x' ||_2

    Args
    ----
    x_original : torch.Tensor
        Original input, any shape.
    x_boundary : torch.Tensor
        Boundary point, same shape as x_original.
    p : float
        Norm to use (2.0 = Euclidean).

    Returns
    -------
    float
        Distance between x_original and x_boundary.
    """
    x_o = x_original.view(-1).detach().cpu()
    x_b = x_boundary.view(-1).detach().cpu()
    return torch.norm(x_o - x_b, p=p).item()


# ----------------------------
# 3. Stability (repeatability)
# ----------------------------

@dataclass
class StabilityResult:
    success_rate: float
    mean_pairwise_distance: float
    std_to_mean: float
    n_success: int
    n_runs: int


def stability_metric(
    model: torch.nn.Module,
    x: torch.Tensor,
    search_fn: BoundarySearchFn,
    n_runs: int = 5,
    device: Optional[torch.device] = None,
) -> StabilityResult:
    """
    Evaluate stability of a decision boundary search.

    Runs the search n_runs times starting from the same x and measures:

    - success_rate: how often a boundary was found
    - mean_pairwise_distance: average L2 distance between all pairs of boundary points
    - std_to_mean: std of distances to the mean boundary point

    Lower distances/std => more stable.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    x = x.to(device)

    boundary_points: List[torch.Tensor] = []
    n_success = 0

    for _ in range(n_runs):
        # ⚠️ search_fn may use autograd (FGSM, IBS, etc.)
        xb, success = search_fn(model, x)
        if success:
            n_success += 1
            boundary_points.append(xb.detach().cpu().view(-1))

    if n_success <= 1:
        return StabilityResult(
            success_rate=n_success / max(n_runs, 1),
            mean_pairwise_distance=float("nan"),
            std_to_mean=float("nan"),
            n_success=n_success,
            n_runs=n_runs,
        )

    stack = torch.stack(boundary_points, dim=0)
    mean_point = stack.mean(dim=0)

    # Pairwise distances
    pairwise_dists = []
    for i in range(stack.size(0)):
        for j in range(i + 1, stack.size(0)):
            pairwise_dists.append(torch.norm(stack[i] - stack[j], p=2).item())
    mean_pairwise = float(torch.tensor(pairwise_dists).mean().item())

    # Distances to mean
    dists_to_mean = torch.norm(stack - mean_point.unsqueeze(0), p=2, dim=1)
    std_to_mean = float(dists_to_mean.std().item())

    return StabilityResult(
        success_rate=n_success / max(n_runs, 1),
        mean_pairwise_distance=mean_pairwise,
        std_to_mean=std_to_mean,
        n_success=n_success,
        n_runs=n_runs,
    )


# ----------------------------
# 4. Flip correctness
# ----------------------------

@torch.no_grad()
def flip_correctness_metric(
    model: torch.nn.Module,
    x_original: torch.Tensor,
    x_boundary: torch.Tensor,
    eps: float = 1e-2,
) -> Dict[str, object]:
    """
    Check whether the found boundary point really lies at a class transition.

    Idea:
    - Determine original predicted class c_orig at x_original.
    - Take x_boundary and move a tiny step +eps and -eps along the line
      between x_original and x_boundary.
    - Evaluate model at those two points and see if the predictions differ.

    Returns a dict with:
      - "flip_correct": bool
      - "class_minus": int
      - "class_plus": int
      - "orig_class": int

    Args
    ----
    model : torch.nn.Module
        Classifier with logits output.
    x_original : torch.Tensor
        Original input (no batch dimension).
    x_boundary : torch.Tensor
        Boundary point (no batch dimension).
    eps : float
        Small step size around the boundary along the line direction.

    Returns
    -------
    dict
    """
    model.eval()

    # Ensure shapes (1, D)
    if x_original.dim() == 1:
        xo = x_original.unsqueeze(0)
    else:
        xo = x_original

    if x_boundary.dim() == 1:
        xb = x_boundary.unsqueeze(0)
    else:
        xb = x_boundary

    device = next(model.parameters()).device
    xo = xo.to(device)
    xb = xb.to(device)

    # Original class
    logits_o = model(xo)
    orig_class = logits_o.argmax(dim=1)[0].item()

    # Direction from original to boundary
    direction = xb - xo
    if torch.norm(direction) < 1e-12:
        # Degenerate case: boundary == original
        return {
            "flip_correct": False,
            "class_minus": orig_class,
            "class_plus": orig_class,
            "orig_class": orig_class,
        }

    unit_dir = direction / torch.norm(direction)

    # Take small steps around the boundary
    x_minus = xb - eps * unit_dir
    x_plus = xb + eps * unit_dir

    logits_minus = model(x_minus)
    logits_plus = model(x_plus)

    class_minus = logits_minus.argmax(dim=1)[0].item()
    class_plus = logits_plus.argmax(dim=1)[0].item()

    flip_correct = class_minus != class_plus

    return {
        "flip_correct": flip_correct,
        "class_minus": class_minus,
        "class_plus": class_plus,
        "orig_class": orig_class,
    }

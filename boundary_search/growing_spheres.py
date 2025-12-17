"""Growing Spheres counterfactual search.

This is a lightweight implementation of the Growing Spheres algorithm
for finding a nearby point that changes the model prediction. The search
expands spherical shells around the input until a flipped class is
found, then refines along the line to the counterfactual.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


@torch.no_grad()
def growing_spheres_search(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    initial_radius: float = 0.1,
    step_radius: float = 0.05,
    max_radius: float = 5.0,
    n_samples: int = 512,
    clamp: Optional[Tuple[float, float]] = (0.0, 1.0),
    refine_steps: int = 15,
) -> Tuple[torch.Tensor, bool]:
    """
    Approximate a counterfactual boundary point using Growing Spheres.

    The algorithm samples points in expanding spherical shells centred at
    ``x`` until the model prediction flips. The closest flipped sample is
    then refined with a binary search towards the original input.

    Args:
        model: Classifier that outputs logits.
        x: Input without batch dimension.
        initial_radius: Starting radius for the outer shell.
        step_radius: Increment added to the outer shell after each failed
            expansion.
        max_radius: Maximum radius explored before giving up.
        n_samples: Number of samples drawn per shell expansion.
        clamp: Optional ``(low, high)`` tuple to clamp sampled points.
        refine_steps: Binary search refinement steps once a flip is found.

    Returns:
        (boundary_point, success)
    """

    def _ensure_batch(sample: torch.Tensor) -> torch.Tensor:
        if sample.dim() == 1:
            return sample.unsqueeze(0)
        if sample.dim() == 3:
            return sample.unsqueeze(0)
        if sample.dim() not in {2, 4}:
            raise ValueError(f"Unsupported input shape {sample.shape}")
        return sample

    model.eval()
    xb = _ensure_batch(x)
    first_param = next(model.parameters(), None)
    device = first_param.device if first_param is not None else torch.device("cpu")
    xb = xb.to(device)
    x_flat = xb.view(1, -1)

    orig_class = model(xb).argmax(dim=1)[0].item()

    def sample_shell(outer_r: float, inner_r: float) -> torch.Tensor:
        dims = x_flat.shape[1]
        noise = torch.randn(n_samples, dims, device=device)
        noise = noise / torch.norm(noise, dim=1, keepdim=True)
        scales = torch.empty(n_samples, 1, device=device).uniform_(inner_r, outer_r)
        points = x_flat + noise * scales
        if clamp is not None:
            low, high = clamp
            points = points.clamp(min=low, max=high)
        return points

    outer_radius = max(initial_radius, 1e-6)
    inner_radius = 0.0
    candidate = None
    success = False

    while outer_radius <= max_radius:
        samples = sample_shell(outer_radius, inner_radius)
        preds = model(samples).argmax(dim=1)
        mask = preds != orig_class
        if mask.any():
            flipped = samples[mask]
            # Choose closest flipped sample to the original
            dists = torch.norm(flipped - x_flat, p=2, dim=1)
            idx = torch.argmin(dists)
            candidate = flipped[idx].view_as(xb)
            success = True
            break

        inner_radius = outer_radius
        outer_radius += step_radius

    if not success:
        return xb.squeeze(0).detach().cpu(), False

    # Binary search refinement between x and candidate
    low_point = xb
    high_point = candidate
    for _ in range(refine_steps):
        mid = 0.5 * (low_point + high_point)
        pred_mid = model(mid).argmax(dim=1)[0].item()
        if pred_mid == orig_class:
            low_point = mid
        else:
            high_point = mid

    boundary = 0.5 * (low_point + high_point)
    if clamp is not None:
        low, high = clamp
        boundary = boundary.clamp(min=low, max=high)
    return boundary.squeeze(0).detach().cpu(), True


def optimize_proximity_with_growing_spheres(
    model: torch.nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    *,
    initial_radius: float = 0.1,
    step_radius: float = 0.05,
    n_samples: int = 512,
    clamp: Optional[Tuple[float, float]] = (0.0, 1.0),
    refine_steps: int = 10,
) -> Tuple[torch.Tensor, bool]:
    """
    Refine a boundary baseline using Growing Spheres to reduce proximity.

    This helper limits the search radius to the distance between ``x`` and the
    provided ``baseline``; if a closer counterfactual is found within that
    radius, it is returned along with ``success=True``. Otherwise, the original
    baseline is returned with ``success=False``.
    """

    # Distance to baseline defines the maximal search budget
    x_flat = x.view(1, -1)
    baseline_flat = baseline.view(1, -1)
    max_radius = torch.norm(baseline_flat - x_flat, p=2).item()

    # Nothing to optimize if the baseline is identical
    if max_radius <= 0:
        return baseline.detach().cpu(), False

    # Ensure we don't start with a radius larger than the allowed budget
    bounded_initial = min(initial_radius, max_radius)
    candidate, success = growing_spheres_search(
        model,
        x,
        initial_radius=bounded_initial,
        step_radius=step_radius,
        max_radius=max_radius,
        n_samples=n_samples,
        clamp=clamp,
        refine_steps=refine_steps,
    )

    if not success:
        return baseline.detach().cpu(), False

    # Only accept strictly closer solutions
    candidate_dist = torch.norm(candidate.view(1, -1) - x_flat.cpu(), p=2).item()
    baseline_dist = max_radius
    if candidate_dist + 1e-9 < baseline_dist:
        return candidate, True

    return baseline.detach().cpu(), False
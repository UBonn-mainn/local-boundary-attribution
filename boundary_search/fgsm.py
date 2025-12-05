import torch
import torch.nn.functional as F
from typing import Optional, Tuple


@torch.no_grad()
def _predict_class(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return predicted class indices for a batch x."""
    logits = model(x)
    return logits.argmax(dim=1)


def fgsm_boundary_search(
    model: torch.nn.Module,
    x: torch.Tensor,
    step_size: float = 0.01,
    max_iters: int = 50,
    clamp: Optional[Tuple[float, float]] = (0.0, 1.0),
    refine_steps: int = 15,
) -> Tuple[torch.Tensor, bool]:
    """
    Approximate a point on the decision boundary starting from x using FGSM-like updates.

    Args
    ----
    model : torch.nn.Module
        Trained classifier. Should output logits of shape (batch, num_classes).
    x : torch.Tensor
        Input sample of shape (C, H, W) or (D,). Must be a single sample (no batch dim).
    step_size : float
        Step size for the FGSM update (epsilon per iteration).
    max_iters : int
        Maximum number of FGSM iterations before giving up if no flip occurs.
    clamp : (float, float) or None
        If not None, clamp inputs into this range after each step (e.g. (0,1) for images).
    refine_steps : int
        Number of binary search steps to refine the boundary point between last
        “same class” and first “flipped class” point.

    Returns
    -------
    boundary_point : torch.Tensor
        Approximate point on the decision boundary, same shape as x.
    success : bool
        True if a class flip was observed and a boundary point found, False otherwise.

    Notes
    -----
    * This is an **untargeted** attack-style search: it tries to leave the original class.
    * If no flip happens within `max_iters`, the function returns the last x and success=False.
    """
    model.eval()

    # Ensure x has batch dimension
    if x.dim() == 1:
        x = x.unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
    elif x.dim() != 2 and x.dim() != 4:
        raise ValueError(f"Unsupported input shape {x.shape}")

    device = next(model.parameters()).device
    x = x.to(device)

    # Original prediction
    with torch.no_grad():
        orig_class = _predict_class(model, x)[0].item()

    # Will track last point with original class and first point with flipped class
    x_curr = x.clone().detach()
    x_prev = x_curr.clone().detach()
    flipped_point = None
    flipped = False

    for it in range(max_iters):
        x_curr.requires_grad_(True)
        model.zero_grad(set_to_none=True)

        logits = model(x_curr)
        # Loss that encourages leaving the original class
        target = torch.tensor([orig_class], device=device)
        loss = F.cross_entropy(logits, target)
        loss.backward()

        # FGSM step: move to increase loss of original class
        with torch.no_grad():
            grad_sign = x_curr.grad.sign()
            x_next = x_curr + step_size * grad_sign

            if clamp is not None:
                low, high = clamp
                x_next = torch.clamp(x_next, low, high)

        # Check prediction after the step
        with torch.no_grad():
            pred_next = _predict_class(model, x_next)[0].item()

        if pred_next != orig_class:
            # We have crossed the decision boundary between x_curr and x_next
            flipped_point = x_next.clone().detach()
            flipped = True
            break
        else:
            # Still same class; move on
            x_prev = x_curr.detach()
            x_curr = x_next.detach()

    if not flipped:
        # No boundary crossing detected
        return x_curr.squeeze(0).detach().cpu(), False

    # ---------- Binary search refinement between x_prev (orig_class) and flipped_point ----------
    low_point = x_prev.clone().detach()
    high_point = flipped_point.clone().detach()

    with torch.no_grad():
        for _ in range(refine_steps):
            mid = 0.5 * (low_point + high_point)
            mid_class = _predict_class(model, mid)[0].item()

            if mid_class == orig_class:
                # boundary is still further towards high_point
                low_point = mid
            else:
                # already flipped, boundary is between low_point and mid
                high_point = mid

    # mid of final segment as approximate boundary point
    boundary_point = 0.5 * (low_point + high_point)
    return boundary_point.squeeze(0).detach().cpu(), True

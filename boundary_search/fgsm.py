import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

@dataclass
class BoundarySearchResult:
    x_start: np.ndarray
    x_boundary: np.ndarray
    x_enemy: Optional[np.ndarray]
    num_steps: int
    success: bool
    meta: Dict[str, Any]


class FGSMBoundarySearch:
    def __init__(
        self,
        model: torch.nn.Module,
        step_size: float = 0.01,
        max_steps: int = 200,
        boundary_bisect_steps: int = 25,
        clamp: Optional[Tuple[float, float]] = None,
        device: Optional[torch.device] = None,
        max_retries: int = 10,
    ):
        self.model = model
        self.step_size = float(step_size)
        self.max_steps = int(max_steps)
        self.boundary_bisect_steps = int(boundary_bisect_steps)
        self.clamp = clamp
        self.device = device or torch.device("cpu")
        self.max_retries = int(max_retries)

        self.model.to(self.device)
        self.model.eval()

        logger.debug(
            "Initialized FGSMBoundarySearch(step_size=%s, max_steps=%s, bisect=%s, clamp=%s, device=%s, max_retries=%s)",
            self.step_size, self.max_steps, self.boundary_bisect_steps, self.clamp, self.device, self.max_retries
        )

    @torch.no_grad()
    def _predict_class(self, x_b: torch.Tensor) -> int:
        logits = self.model(x_b)
        return int(torch.argmax(logits, dim=-1).item())

    def _label(self, x_np: np.ndarray) -> int:
        x_t = torch.tensor(x_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self._predict_class(x_t)

    def _bisect_to_boundary(self, x0: np.ndarray, x1: np.ndarray, y0: int) -> np.ndarray:
        lo, hi = x0.copy(), x1.copy()
        for k in range(self.boundary_bisect_steps):
            mid = 0.5 * (lo + hi)
            if self._label(mid) == y0:
                lo = mid
            else:
                hi = mid
            logger.debug("FGSM bisection step %d/%d", k + 1, self.boundary_bisect_steps)
        return 0.5 * (lo + hi)

    def search(self, x: np.ndarray, y: Optional[int] = None) -> BoundarySearchResult:
        x = np.asarray(x, dtype=np.float32)
        y0 = int(y) if y is not None else self._label(x)

        logger.debug("FGSM search start (y0=%s, dim=%s)", y0, x.shape)

        x_t = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        success = False
        x_enemy_np = None
        steps_taken = 0
        final_x_boundary = x.copy()
        
        current_step_size = self.step_size

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                logger.info(f"FGSM retry {attempt}/{self.max_retries}: doubling step size to {current_step_size:.4f}")
            
            x_adv = x_t.clone().detach().requires_grad_(True)
            flipped = False
            
            for step in range(self.max_steps):
                steps_taken = step + 1
                logits = self.model(x_adv)
                loss = F.cross_entropy(logits, torch.tensor([y0], device=self.device))

                self.model.zero_grad(set_to_none=True)
                if x_adv.grad is not None:
                    x_adv.grad.zero_()
                loss.backward()

                with torch.no_grad():
                    # Handle zero grad case
                    if x_adv.grad is None:
                         grad_sign = torch.zeros_like(x_adv)
                    else:
                         grad_sign = x_adv.grad.sign()
                         
                    x_next = x_adv + current_step_size * grad_sign
                    if self.clamp is not None:
                        lo, hi = self.clamp
                        x_next = torch.clamp(x_next, lo, hi)

                y_next = self._predict_class(x_next)
                logger.debug("FGSM iter %d/%d: pred=%s loss=%.6f", steps_taken, self.max_steps, y_next, float(loss.item()))

                if y_next != y0:
                    success = True
                    flipped = True
                    x_enemy_np = x_next.detach().squeeze(0).cpu().numpy()
                    logger.info("FGSM boundary crossed at iter %d (y0=%s -> y=%s)", steps_taken, y0, y_next)
                    break

                x_adv = x_next.detach().requires_grad_(True)
            
            if flipped:
                break
                
            # If not flipped, loop continues with doubled step size
            current_step_size *= 2.0

        if not success:
            logger.warning("FGSM failed to cross boundary after %d retries (max_steps=%d, y0=%s)", self.max_retries, self.max_steps, y0)
            final_x_boundary = x.copy()
        else:
            final_x_boundary = self._bisect_to_boundary(x, x_enemy_np, y0)
            logger.info("FGSM boundary refined with %d bisection steps", self.boundary_bisect_steps)

        return BoundarySearchResult(
            x_start=x,
            x_boundary=final_x_boundary,
            x_enemy=x_enemy_np,
            num_steps=steps_taken,
            success=success,
            meta={
                "method": "fgsm",
                "step_size": self.step_size,
                "final_step_size": current_step_size, 
                "max_steps": self.max_steps,
                "bisect_steps": self.boundary_bisect_steps,
                "clamp": self.clamp,
                "retries": attempt
            },
        )


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
    max_retries: int = 10,
) -> Tuple[torch.Tensor, bool]:
    """
    Approximate a point on the decision boundary starting from x using FGSM-like updates.
    Includes adaptive step size to handle different data scales.

    Args
    ----
    model : torch.nn.Module
        Trained classifier. Should output logits of shape (batch, num_classes).
    x : torch.Tensor
        Input sample of shape (C, H, W) or (D,). Must be a single sample (no batch dim).
    step_size : float
        Initial step size for the FGSM update.
    max_iters : int
        Maximum number of FGSM iterations per try.
    clamp : (float, float) or None
        If not None, clamp inputs into this range after each step (e.g. (0,1) for images).
    refine_steps : int
        Number of binary search steps to refine the boundary point.
    max_retries : int
        Number of times to retry with doubled step size if no boundary is found.

    Returns
    -------
    boundary_point : torch.Tensor
        Approximate point on the decision boundary, same shape as x.
    success : bool
        True if a class flip was observed and a boundary point found, False otherwise.
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
        orig_cls_tensor = _predict_class(model, x)
        orig_class = orig_cls_tensor[0].item()

    current_step_size = step_size
    
    # Adaptive loop
    for attempt in range(max_retries + 1):
        if attempt > 0:
            logger.info(f"FGSM retry {attempt}/{max_retries}: doubling step size to {current_step_size:.4f}")

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
                if x_curr.grad is None:
                    # Very flat area or zero grad?
                    grad_sign = torch.zeros_like(x_curr)
                else:
                    grad_sign = x_curr.grad.sign()
                
                x_next = x_curr + current_step_size * grad_sign

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

        if flipped:
            # Success! Found a crossing.
            break
        
        # If not flipped, we loop again with larger step_size
        current_step_size *= 2.0

    if not flipped:
        # No boundary crossing detected after all retries
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


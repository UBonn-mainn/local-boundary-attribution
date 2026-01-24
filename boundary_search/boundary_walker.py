
import logging
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from .fgsm import FGSMBoundarySearch
from .ibs import BoundarySearchResult

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Tolerance thresholds
NEWTON_CONVERGENCE_TOL: float = 1e-3
GRADIENT_VANISH_TOL: float = 1e-8
TANGENT_ZERO_TOL: float = 1e-6
MIN_STEP_SIZE: float = 1e-5

# Step size multipliers for adaptive stepping
STEP_MULTIPLIERS: Tuple[float, ...] = (1.0, 2.0)

# Multipliers for finding boundary crossing points
BOUNDARY_CROSSING_MULTIPLIERS: Tuple[float, ...] = (1.5, 2.0, 3.0, 5.0)


class BoundaryCrawler:
    """
    Iterative Boundary Crawler using gradient-based manifold descent.
    
    Algorithm:
    1. Start with an initial boundary point (found via FGSM).
    2. Iteratively crawl along the decision boundary surface toward the original input.
    3. Use Newton-Raphson for efficient boundary restoration after each step.
    
    Runtime Complexity: O(Iterations * (2 Forward + 1 Backward + BisectionSteps))
    """
    
    def __init__(
        self, 
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        fgsm_params: Optional[Dict[str, Any]] = None,
        crawl_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.fgsm_params = fgsm_params or {}
        
        params = crawl_params or {}
        self.max_iterations: int = params.get("max_iterations", 10)
        self.step_size: float = params.get("step_size", 0.05)
        self.bisection_steps: int = params.get("bisection_steps", 10)
        
        # Multi-start FGSM parameters
        self.n_fgsm_starts: int = params.get("n_fgsm_starts", 5)
        self.perturbation_radius: float = params.get("perturbation_radius", 0.05)
        
        # Random perturbation escape parameters
        self.stuck_threshold: int = params.get("stuck_threshold", 2)
        
        self.fgsm_searcher = FGSMBoundarySearch(
            model=model, 
            device=self.device, 
            **self.fgsm_params
        )

    def _to_device(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on the correct device."""
        if x.device != self.device:
            return x.to(self.device)
        return x

    def _predict(self, x: torch.Tensor) -> int:
        """Get model prediction for a tensor."""
        with torch.no_grad():
            return self.model(x).argmax(dim=-1).item()

    def _newton_restore(self, x_in: torch.Tensor, y_orig: int, max_steps: int = 5, tol: float = NEWTON_CONVERGENCE_TOL) -> Optional[torch.Tensor]:
        """
        Uses Newton's method to find the boundary (root of Logit_A - Logit_B = 0).
        More query-efficient than bisection (approx 2-3 steps vs 10-15).
        """
        x = self._to_device(x_in.clone().detach())
        
        for _ in range(max_steps):
            x.requires_grad_(True)
            with torch.enable_grad():
                logits = self.model(x)
                
                probs = F.softmax(logits, dim=-1)
                top2 = torch.topk(probs, 2, dim=-1).indices[0]
                
                c1, c2 = top2[0].item(), top2[1].item()
                
                if c1 == y_orig:
                    y_enemy = c2
                else:
                    y_enemy = c1
                    
                score = logits[0, y_orig] - logits[0, y_enemy]
                
                # Check convergence
                if abs(score.item()) < tol:
                    return x.detach()
                    
                # Newton Step: x_new = x - f(x)/||grad||^2 * grad
                # Gradient of score
                self.model.zero_grad()
                score.backward()
                
                g = x.grad
                g_norm_sq = torch.sum(g**2).item()
                
                if g_norm_sq < GRADIENT_VANISH_TOL:
                    return None  # Gradient vanished
                
                # Update
                step = (score.item() / g_norm_sq) * g
                x = x - step
                x = x.detach()
                
        # Final check
        logits = self.model(x)
        probs = F.softmax(logits, dim=-1)
        top2 = torch.topk(probs, 2, dim=-1).indices[0]
        
        if y_orig not in top2.tolist():
             return None # Diverged too far
             
        return x.detach()

    def _bisect(self, x1: torch.Tensor, x2: torch.Tensor, y_ref: int) -> Optional[torch.Tensor]:
        """
        Bisect between x1 (class y_ref) and x2 (class != y_ref) to find boundary.
        Returns None if both have the same class.
        """
        x1 = self._to_device(x1)
        x2 = self._to_device(x2)
        
        pred1 = self._predict(x1)
        pred2 = self._predict(x2)
        
        if pred1 == pred2:
            return None
        
        low = x1
        high = x2
        
        with torch.no_grad():
            for _ in range(self.bisection_steps):
                mid = (low + high) / 2
                pred_mid = self._predict(mid)
                if pred_mid == y_ref:
                    low = mid
                else:
                    high = mid
                
        return (low + high) / 2

    def _get_boundary_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the normal vector of the decision boundary at x.
        Normal is gradient of (Logit_A - Logit_B).
        """
        x_in = self._to_device(x.clone().detach()).requires_grad_(True)
        logits = self.model(x_in)
        
        probs = F.softmax(logits, dim=-1)
        top2 = torch.topk(probs, 2, dim=-1).indices[0] # [c1, c2]
        c1, c2 = top2[0], top2[1]
        
        # Calculate score difference
        score = logits[0, c1] - logits[0, c2]
        
        # Backprop
        self.model.zero_grad()
        score.backward()
        
        grad = x_in.grad.detach()
        # Visualize: The gradient points towards Class C1 (higher score).
        # We normalize it.
        return F.normalize(grad, p=2, dim=-1)

    def _projected_step(self, x_current: torch.Tensor, x_orig: torch.Tensor, step_size: Optional[float] = None) -> Optional[torch.Tensor]:
        """
        Take a step along the boundary surface using gradient projection onto tangent plane.
        """
        # 1. Get Normal Vector n
        n = self._get_boundary_gradient(x_current)
        
        # 2. Vector towards original point v
        v = x_orig - x_current
        
        # 3. Project v onto tangent plane
        # v_tan = v - (v . n) * n
        dot = torch.sum(v * n, dim=-1, keepdim=True)
        v_tan = v - dot * n
        
        # Normalize tangent direction
        if torch.norm(v_tan) < TANGENT_ZERO_TOL:
            logger.debug("Crawler: Already at optimal projection (tangent is zero).")
            return None
            
        v_dir = F.normalize(v_tan, p=2, dim=-1)
        
        # 4. Take step in tangent direction
        # We move by 'step_size' along this optimal path
        s = step_size if step_size is not None else self.step_size
        x_candidate_out = x_current + s * v_dir
        
        return x_candidate_out

    def _multi_start_fgsm_init(self, x: np.ndarray, y: Optional[int] = None) -> List[BoundarySearchResult]:
        """
        Run FGSM from multiple perturbed starting points and return all successful results.
        """
        candidates = []
        
        # Try from original point
        res_orig = self.fgsm_searcher.search(x, y=y)
        if res_orig.success:
            candidates.append(res_orig)
        
        # Try from perturbed starting points
        for _ in range(self.n_fgsm_starts - 1):
            noise = np.random.randn(*x.shape) * self.perturbation_radius
            x_perturbed = x + noise
            res = self.fgsm_searcher.search(x_perturbed, y=y)
            if res.success:
                candidates.append(res)
        
        return candidates

    def _find_boundary_crossing(self, x_orig: torch.Tensor, candidate: torch.Tensor, y_orig: int) -> Optional[torch.Tensor]:
        """
        Find a point that crosses the decision boundary along the direction from x_orig through candidate.
        Returns None if no crossing is found within the search multipliers.
        """
        pred_cand = self._predict(candidate)
        
        if pred_cand != y_orig:
            return candidate
        
        # Try extending in the same direction
        diff = candidate - x_orig
        for mult in BOUNDARY_CROSSING_MULTIPLIERS:
            c_out = x_orig + diff * mult
            if self._predict(c_out) != y_orig:
                return c_out
        
        return None

    def _restore_to_boundary(self, candidate: torch.Tensor, x_orig: torch.Tensor, y_orig: int) -> Optional[torch.Tensor]:
        """
        Restore a candidate point to the decision boundary.
        Tries Newton-Raphson first, falls back to bisection if that fails.
        """
        # Try Newton-Raphson first (fast)
        restored = self._newton_restore(candidate, y_orig)
        if restored is not None:
            return restored
        
        # Fallback to bisection
        boundary_crossing = self._find_boundary_crossing(x_orig, candidate, y_orig)
        if boundary_crossing is not None:
            return self._bisect(x_orig, boundary_crossing, y_orig)
        
        return None

    def search(self, x: np.ndarray, y: Optional[int] = None) -> BoundarySearchResult:
        """
        Search for the closest boundary point to input x.
        
        Args:
            x: Input point as numpy array.
            y: Optional true label. If None, inferred from model prediction.
            
        Returns:
            BoundarySearchResult with the closest boundary point found.
        """
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Get original label for reference
        if y is None:
            y_orig = self._predict(x_tensor)
        else:
            y_orig = y
        
        # 1. Multi-start FGSM initialization
        fgsm_candidates = self._multi_start_fgsm_init(x, y=y)
        
        if not fgsm_candidates:
            logger.warning("Crawler: All FGSM initializations failed.")
            return BoundarySearchResult(
                x_start=x,
                x_boundary=x,
                x_enemy=x,
                num_steps=0,
                success=False,
                meta={"method": "boundary_crawler", "error": "FGSM init failed"}
            )
        
        # Pick closest to original x
        best_init = min(fgsm_candidates, key=lambda r: np.linalg.norm(r.x_boundary - x))
        current_boundary = torch.tensor(best_init.x_boundary, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        best_dist = torch.norm(current_boundary - x_tensor).item()
        logger.debug(f"Crawler Start: Best of {len(fgsm_candidates)} FGSM starts, dist={best_dist:.4f}")
        
        total_steps = 0
        current_step_size = self.step_size
        no_improve_count = 0
        
        for i in range(self.max_iterations):
            refined_candidates = []
            
            # 2. Adaptive step sizes: try multiple step multipliers
            for step_mult in STEP_MULTIPLIERS:
                s = current_step_size * step_mult
                cand_step = self._projected_step(current_boundary, x_tensor, step_size=s)
                
                if cand_step is None:
                    continue
                
                # Restore to boundary (Newton + bisection fallback)
                cand_refined = self._restore_to_boundary(cand_step, x_tensor, y_orig)
                if cand_refined is not None:
                    refined_candidates.append(cand_refined)
            
            # Check for all-None steps (optimal reached)
            if not refined_candidates:
                logger.debug(f"Crawler Iter {i}: No valid candidates. Stopping.")
                break
            
            step_improved = False
            
            for cand in refined_candidates:
                dist = torch.norm(cand - x_tensor).item()
                if dist < best_dist:
                    best_dist = dist
                    current_boundary = cand
                    step_improved = True
            
            total_steps += 1
            
            if step_improved:
                logger.debug(f"Crawler Iter {i}: Improved dist to {best_dist:.4f}")
                no_improve_count = 0
            else:
                no_improve_count += 1
                
                # 3. Random perturbation to escape local minima
                if no_improve_count >= self.stuck_threshold:
                    logger.debug(f"Crawler Iter {i}: Stuck for {no_improve_count} iters. Applying random perturbation.")
                    noise = torch.randn_like(current_boundary) * self.perturbation_radius
                    perturbed = current_boundary + noise
                    
                    # Try to restore to boundary
                    restored = self._restore_to_boundary(perturbed, x_tensor, y_orig)
                    
                    if restored is not None:
                        current_boundary = restored
                        no_improve_count = 0
                        logger.debug(f"Crawler Iter {i}: Perturbation applied successfully.")
                    else:
                        # Reduce step size if perturbation failed
                        current_step_size *= 0.5
                else:
                    current_step_size *= 0.5
                    logger.debug(f"Crawler Iter {i}: No improvement. Reducing step to {current_step_size:.4f}")
                
                if current_step_size < MIN_STEP_SIZE:
                    logger.debug("Crawler: Step size too small. Stopping.")
                    break
                
        x_final = current_boundary.squeeze(0).cpu().numpy()
        
        return BoundarySearchResult(
            x_start=x,
            x_boundary=x_final,
            x_enemy=x_final,
            num_steps=total_steps,
            success=True,
            meta={
                "method": "boundary_crawler",
                "iterations": i + 1,
                "final_dist": best_dist,
                "n_fgsm_starts": len(fgsm_candidates)
            }
        )

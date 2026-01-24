
import logging
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from .fgsm import FGSMBoundarySearch
from .ibs import BoundarySearchResult

logger = logging.getLogger(__name__)


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
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.fgsm_params = fgsm_params or {}
        
        params = crawl_params or {}
        self.max_iterations = params.get("max_iterations", 10)
        self.step_size = params.get("step_size", 0.05)
        self.bisection_steps = params.get("bisection_steps", 10)
        
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

    def _newton_restore(self, x_in: torch.Tensor, y_orig: int, max_steps: int = 5, tol: float = 1e-3) -> Optional[torch.Tensor]:
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
                
                if g_norm_sq < 1e-8:
                    return None # Gradient vanished
                
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
        
        with torch.no_grad():
            logits1 = self.model(x1)
            pred1 = logits1.argmax(dim=-1).item()
            
            logits2 = self.model(x2)
            pred2 = logits2.argmax(dim=-1).item()
        
        if pred1 == pred2:
            return None
        
        low = x1
        high = x2
        
        with torch.no_grad():
            for _ in range(self.bisection_steps):
                mid = (low + high) / 2
                pred_mid = self.model(mid).argmax(dim=-1).item()
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
        if torch.norm(v_tan) < 1e-6:
            logger.debug("Crawler: Already at optimal projection (tangent is zero).")
            return None
            
        v_dir = F.normalize(v_tan, p=2, dim=-1)
        
        # 4. Take step in tangent direction
        # We move by 'step_size' along this optimal path
        s = step_size if step_size is not None else self.step_size
        x_candidate_out = x_current + s * v_dir
        
        return x_candidate_out

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
        
        # 1. Initial Point via FGSM
        res_fgsm = self.fgsm_searcher.search(x, y=y)
        if not res_fgsm.success:
            logger.warning("Crawler: FGSM initialization failed.")
            return res_fgsm
            
        current_boundary = torch.tensor(res_fgsm.x_boundary, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Get original label for reference
        if y is None:
            with torch.no_grad():
                y_orig = self.model(x_tensor).argmax(dim=-1).item()
        else:
            y_orig = y
        
        best_dist = torch.norm(current_boundary - x_tensor).item()
        logger.debug(f"Crawler Start: Dist {best_dist:.4f}")
        
        total_steps = 0
        current_step_size = self.step_size
        
        for i in range(self.max_iterations):
            refined_candidates = []
            
            cand_step = self._projected_step(current_boundary, x_tensor, step_size=current_step_size)
            
            if cand_step is None:
                break # Optimal reached
            
            # 2. Refine using Newton-Raphson (Fast)
            cand_refined = self._newton_restore(cand_step, y_orig)
            
            if cand_refined is not None:
                refined_candidates.append(cand_refined)
            else:
                with torch.no_grad():
                    pred_cand = self.model(cand_step).argmax(dim=-1).item()
                candidate_to_bisect = None
                
                if pred_cand != y_orig:
                    candidate_to_bisect = cand_step
                else:
                    diff = cand_step - x_tensor
                    for mult in [1.5, 2.0, 3.0, 5.0]:
                        c_out = x_tensor + diff * mult
                        with torch.no_grad():
                            if self.model(c_out).argmax(dim=-1).item() != y_orig:
                                candidate_to_bisect = c_out
                                break
                
                if candidate_to_bisect is not None:
                    b = self._bisect(x_tensor, candidate_to_bisect, y_orig)
                    if b is not None:
                        refined_candidates.append(b)

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
            else:
                current_step_size *= 0.5
                logger.debug(f"Crawler Iter {i}: No improvement. Reducing step to {current_step_size:.4f}")
                if current_step_size < 1e-4:
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
                "final_dist": best_dist
            }
        )


import logging
from typing import Optional, Dict, Any, Union, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from .fgsm import FGSMBoundarySearch
from .ibs import IBSBoundarySearch, BoundarySearchResult

logger = logging.getLogger(__name__)

class BoundaryCrawler:
    """
    Iterative Boundary Crawler (Crawler).
    
    Iterative Boundary Crawler (Crawler).
    
    Algorithm:
    1. Start with an initial boundary point (found via FGSM).
    2. Iteratively 'crawl' along the decision boundary surface to find a point 
       closer to the original input x.
       
    The crawling can happen in two modes:
    A) Random Ring (Default):
       - Sample discrete points in a local 'ring' (hypersphere) tangent to the surface.
       - Project them onto the boundary using bisection.
       - Pick the best one. Inefficient in High-D.
       
    B) Gradient Descent (Manifold Descent):
       - Calculate the exact Normal Vector of the boundary using autograd.
       - Calculate the 'ideal' movement direction: the vector pointing towards x_orig,
         PROJECTED onto the tangent plane of the boundary.
       - Take a step in this optimal direction and re-project to the boundary.
    
    Runtime Complexity:
    - Random: O(Iterations * Samples * BisectionSteps)
    - Gradient: O(Iterations * (2 Forward + 1 Backward + BisectionSteps))
    """
    def __init__(
        self, 
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        fgsm_params: Optional[Dict[str, Any]] = None,
        crawl_params: Optional[Dict[str, Any]] = None,
    ):
        self.device = device or torch.device("cpu")
        self.model = model
        self.fgsm_params = fgsm_params or {}
        
        # Crawl parameters
        params = crawl_params or {}
        self.mode = params.get("mode", "random") # 'random' or 'gradient'
        self.max_iterations = params.get("max_iterations", 10)
        self.num_samples = params.get("num_samples", 10) # 'K' samples on the ring (Random Mode)
        self.step_size = params.get("step_size", 0.05)   # Step size / Radius
        self.bisection_steps = params.get("bisection_steps", 10)
        
        self.fgsm_searcher = FGSMBoundarySearch(
            model=model, 
            device=self.device, 
            **self.fgsm_params
        )

    def _newton_restore(self, x_in: torch.Tensor, y_orig: int, max_steps=5, tol=1e-3) -> Optional[torch.Tensor]:
        """
        Uses Newton's method to find the boundary (root of Logit_A - Logit_B = 0).
        Much more query-efficient than bisection (approx 2-3 steps vs 10-15).
        """
        x = x_in.clone().detach()
        
        for k in range(max_steps):
            x.requires_grad_(True)
            with torch.enable_grad():
                logits = self.model(x)
                
                # Identify the relevant "enemy" class
                # We define the boundary function f(x) = Logit(y_orig) - Logit(y_enemy)
                # We want f(x) = 0.
                
                probs = F.softmax(logits, dim=-1)
                top2 = torch.topk(probs, 2, dim=-1).indices[0] # [c1, c2]
                
                c1, c2 = top2[0].item(), top2[1].item()
                
                if c1 == y_orig:
                    y_enemy = c2
                else:
                    y_enemy = c1 # Use the top class as enemy if we are outside
                    
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
        pred = logits.argmax(dim=-1).item()
        
        # We accept if we are VERY close to boundary, or just flipped? 
        # Actually Newton ensures f(x) ~ 0.
        # But for 'boundary' strictly we usually want a point ON the boundary
        # or maybe a pair (A, B). 
        # For this crawler, 'current_boundary' is just a point very close to the decision surface.
        
        # Let's verify we didn't diverge to a totally different class
        probs = F.softmax(logits, dim=-1)
        top2 = torch.topk(probs, 2, dim=-1).indices[0]
        if y_orig not in top2.tolist():
             return None # Diverged too far
             
        # Return the refined point
        return x.detach()

    def _bisect(self, x1: torch.Tensor, x2: torch.Tensor, y_ref: int) -> Optional[torch.Tensor]:
        """
        Bisect between x1 (class y_ref) and x2 (class != y_ref) to find boundary.
        Returns None if both are same class (assumption violated).
        """
        # Quick check endpoints
        logits1 = self.model(x1)
        pred1 = logits1.argmax(dim=-1).item()
        
        logits2 = self.model(x2)
        pred2 = logits2.argmax(dim=-1).item()
        
        if pred1 == pred2:
            return None
        
        low = x1
        high = x2
        
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
        Normal ~ Gradient of (Logit_A - Logit_B).
        """
        x_in = x.clone().detach().requires_grad_(True)
        logits = self.model(x_in)
        
        # We need the gradient of the 'boundary score'.
        # At boundary between class A and B: Score = LogitA - LogitB = 0.
        # Normal is grad(LogitA - LogitB).
        
        # Identify top 2 classes
        # (For binary, just Logit1 - Logit0)
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
        Take a step along the boundary surface using gradient projection.
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
            logger.debug("Gradient Crawler: Already at optimal projection (tangent is zero).")
            return None
            
        v_dir = F.normalize(v_tan, p=2, dim=-1)
        
        # 4. Take step in tangent direction
        # We move by 'step_size' along this optimal path
        s = step_size if step_size is not None else self.step_size
        x_candidate_out = x_current + s * v_dir
        
        return x_candidate_out

    def search(self, x: np.ndarray) -> BoundarySearchResult:
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # 1. Initial Point via FGSM
        res_fgsm = self.fgsm_searcher.search(x)
        if not res_fgsm.success:
            logger.warning("Crawler: FGSM initialization failed.")
            return res_fgsm
            
        current_boundary = torch.tensor(res_fgsm.x_boundary, device=self.device).unsqueeze(0)
        
        # Get original label for reference
        y_orig = self.model(x_tensor).argmax(dim=-1).item()
        
        best_dist = torch.norm(current_boundary - x_tensor).item()
        logger.debug(f"Crawler Start: Dist {best_dist:.4f}")
        
        total_steps = 0
        current_step_size = self.step_size
        
        # 2. Iterative Crawling
        for i in range(self.max_iterations):
            
            refined_candidates = []
            
            if self.mode == "gradient":
                # --- GRADIENT MODE ---
                # 1. Calculate ONE optimal tangent step
                cand_step = self._projected_step(current_boundary, x_tensor, step_size=current_step_size)
                
                if cand_step is None:
                    break # Optimal reached
                
                # 2. Refine using Newton-Raphson (Fast)
                cand_refined = self._newton_restore(cand_step, y_orig)
                
                if cand_refined is not None:
                     refined_candidates.append(cand_refined)
                else:
                    # Fallback: Newton failed (diverged/vanished). Try robust Bisection.
                    # We need a valid crossing to bisect.
                    pred_cand = self.model(cand_step).argmax(dim=-1).item()
                    candidate_to_bisect = None
                    
                    if pred_cand != y_orig:
                        candidate_to_bisect = cand_step
                    else: 
                        # Try Push Out
                         diff = cand_step - x_tensor
                         for mult in [1.5, 2.0, 3.0, 5.0]:
                             c_out = x_tensor + diff * mult
                             if self.model(c_out).argmax(dim=-1).item() != y_orig:
                                 candidate_to_bisect = c_out
                                 break
                    
                    if candidate_to_bisect is not None:
                         b = self._bisect(x_tensor, candidate_to_bisect, y_orig)
                         if b is not None:
                             refined_candidates.append(b)

            else:
                # --- RANDOM RING MODE (Legacy) ---
                # Sample a 'ring' (hypersphere shell) around current_boundary
                noise = torch.randn(self.num_samples, *x_tensor.shape[1:], device=self.device)
                noise = F.normalize(noise, p=2, dim=-1)
                ring_points = current_boundary + self.step_size * noise
                
                logits_ring = self.model(ring_points)
                preds_ring = logits_ring.argmax(dim=-1)
                
                for j in range(self.num_samples):
                    p_ring = ring_points[j:j+1]
                    pred = preds_ring[j].item()
                    
                    if pred != y_orig:
                         b = self._bisect(x_tensor, p_ring, y_orig)
                         if b is not None:
                             refined_candidates.append(b)

            # 4. Selection & Adaptation
            step_improved = False
            best_cand_iter = None
            
            for cand in refined_candidates:
                dist = torch.norm(cand - x_tensor).item()
                if dist < best_dist: # STRICT improvement
                    best_dist = dist
                    current_boundary = cand
                    best_cand_iter = cand
                    step_improved = True
            
            if self.mode == "gradient":
                 total_steps += 1
                 if step_improved:
                     logger.debug(f"Crawler Iter {i} (gradient): Improved dist to {best_dist:.4f}")
                     # Optional: Reset or increase step size?
                     # current_step_size = max(current_step_size, self.step_size)
                 else:
                     # Adaptive: Reduce step size and retry
                     current_step_size *= 0.5
                     logger.debug(f"Crawler Iter {i} (gradient): No impr. Reducing step to {current_step_size:.4f}")
                     if current_step_size < 1e-4:
                         logger.debug("Crawler: Step size too small. Stop.")
                         break
                     # Do not break, continue loop to retry
            else:
                 # Random mode
                 total_steps += len(refined_candidates)
                 if step_improved:
                    logger.debug(f"Crawler Iter {i} (random): Improved dist to {best_dist:.4f}")
                 else:
                    logger.debug(f"Crawler Iter {i} (random): No improvement. Stopping.")
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
                "iterations": i+1,
                "final_dist": best_dist
            }
        )

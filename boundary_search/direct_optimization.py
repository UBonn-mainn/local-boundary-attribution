"""
Direct Optimization Boundary Search

Uses FGSM for initialization, then gradient descent on a combined loss:
    min ||b - x||² + λ * (logit_A - logit_B)²

This minimizes distance to x while staying on the decision boundary.
"""

import logging
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from .fgsm import FGSMBoundarySearch
from .ibs import BoundarySearchResult

logger = logging.getLogger(__name__)


class DirectOptimizationSearch:
    """
    Boundary search using direct gradient descent on distance + boundary constraint.
    
    Algorithm:
    1. Find initial boundary point via FGSM (with multi-start)
    2. Optimize: min ||b - x||² + λ·(logit_A - logit_B)²
    
    The λ parameter balances staying on the boundary vs. getting closer to x.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        fgsm_params: Optional[Dict[str, Any]] = None,
        opt_params: Optional[Dict[str, Any]] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.fgsm_params = fgsm_params or {}
        
        params = opt_params or {}
        self.lr = params.get("lr", 0.01)
        self.steps = params.get("steps", 100)
        self.lambda_boundary = params.get("lambda_boundary", 100.0)
        
        # Multi-start FGSM
        self.n_fgsm_starts = params.get("n_fgsm_starts", 5)
        self.perturbation_radius = params.get("perturbation_radius", 0.05)
        
        self.fgsm_searcher = FGSMBoundarySearch(
            model=model,
            device=self.device,
            **self.fgsm_params
        )
    
    def _get_enemy_class(self, logits: torch.Tensor, y_orig: int) -> int:
        """Get the enemy class (highest logit that's not y_orig)."""
        probs = F.softmax(logits, dim=-1)
        top2 = torch.topk(probs, 2, dim=-1).indices[0]
        if top2[0].item() == y_orig:
            return top2[1].item()
        return top2[0].item()
    
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
        
        # Get original label
        if y is None:
            with torch.no_grad():
                y_orig = self.model(x_tensor).argmax(dim=-1).item()
        else:
            y_orig = y
        
        # 1. Multi-start FGSM initialization
        fgsm_candidates = []
        
        # Try from original point
        res_orig = self.fgsm_searcher.search(x, y=y)
        if res_orig.success:
            fgsm_candidates.append(res_orig)
        
        # Try from perturbed starting points
        for _ in range(self.n_fgsm_starts - 1):
            noise = np.random.randn(*x.shape) * self.perturbation_radius
            x_perturbed = x + noise
            res = self.fgsm_searcher.search(x_perturbed, y=y)
            if res.success:
                fgsm_candidates.append(res)
        
        if not fgsm_candidates:
            logger.warning("DirectOpt: All FGSM initializations failed.")
            return BoundarySearchResult(
                x_start=x,
                x_boundary=x,
                x_enemy=x,
                num_steps=0,
                success=False,
                meta={"method": "direct_optimization", "error": "FGSM init failed"}
            )
        
        # Pick closest to original x
        best_init = min(fgsm_candidates, key=lambda r: np.linalg.norm(r.x_boundary - x))
        init_dist = np.linalg.norm(best_init.x_boundary - x)
        logger.debug(f"DirectOpt: Best of {len(fgsm_candidates)} FGSM starts, dist={init_dist:.4f}")
        
        # 2. Direct optimization
        b = torch.tensor(best_init.x_boundary, dtype=torch.float32, device=self.device).unsqueeze(0)
        b = b.clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([b], lr=self.lr)
        
        best_b = b.detach().clone()
        best_dist = init_dist
        
        for step in range(self.steps):
            optimizer.zero_grad()
            
            # Distance loss: want b close to x
            dist_loss = torch.norm(b - x_tensor) ** 2
            
            # Boundary constraint loss: want logit difference = 0
            logits = self.model(b)
            y_enemy = self._get_enemy_class(logits, y_orig)
            boundary_loss = (logits[0, y_orig] - logits[0, y_enemy]) ** 2
            
            # Combined loss
            loss = dist_loss + self.lambda_boundary * boundary_loss
            loss.backward()
            optimizer.step()
            
            # Track best point (on boundary and closest to x)
            with torch.no_grad():
                current_dist = torch.norm(b - x_tensor).item()
                logits_check = self.model(b)
                pred = logits_check.argmax(dim=-1).item()
                
                # Only consider if on or near boundary (top 2 classes are close)
                top2_logits = torch.topk(logits_check, 2, dim=-1).values[0]
                logit_diff = abs(top2_logits[0] - top2_logits[1]).item()
                
                if logit_diff < 1.0 and current_dist < best_dist:
                    best_dist = current_dist
                    best_b = b.detach().clone()
            
            if step % 20 == 0:
                logger.debug(f"DirectOpt Step {step}: dist={current_dist:.4f}, boundary_loss={boundary_loss.item():.4f}")
        
        x_final = best_b.squeeze(0).cpu().numpy()
        final_dist = np.linalg.norm(x_final - x)
        
        return BoundarySearchResult(
            x_start=x,
            x_boundary=x_final,
            x_enemy=x_final,
            num_steps=self.steps,
            success=True,
            meta={
                "method": "direct_optimization",
                "init_dist": init_dist,
                "final_dist": final_dist,
                "improvement": init_dist - final_dist,
                "n_fgsm_starts": len(fgsm_candidates)
            }
        )

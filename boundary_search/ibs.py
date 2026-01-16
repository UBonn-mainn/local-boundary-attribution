
import logging
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .fgsm import BoundarySearchResult, FGSMBoundarySearch

logger = logging.getLogger(__name__)

class IBSBoundarySearch:
    """
    Informed Baseline Search (IBS) algorithm.
    
    Refines the boundary search by using data-driven directions (towards target class samples)
    and precise iterative convergence to the decision boundary.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        X_train: Union[np.ndarray, torch.Tensor],
        y_train: Union[np.ndarray, torch.Tensor],
        num_candidates: int = 10,
        max_iterations: int = 50,
        precision_limit: float = 0.001,
        scaling_factor: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: Trained classifier.
            X_train: Training data features (for sampling target candidates).
            y_train: Training data labels.
            num_candidates: Number of target samples to use for search directions (K).
            max_iterations: Max steps for the iterative root finding.
            precision_limit: Convergence threshold for |P(x) - 0.5|.
            scaling_factor: Factor for adaptive step sizing (not strictly used if bisection is used, but kept for param compatibility).
            device: Torch device.
        """
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Store data for sampling
        if isinstance(X_train, np.ndarray):
            self.X_train = torch.from_numpy(X_train).float()
            self.y_train = torch.from_numpy(y_train).long()
        else:
            self.X_train = X_train.float()
            self.y_train = y_train.long()
            
        self.num_candidates = num_candidates
        self.max_iterations = max_iterations
        self.precision_limit = precision_limit
        
        logger.debug("Initialized IBSBoundarySearch with K=%d candidates", num_candidates)

    @torch.no_grad()
    def _predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns max probability (or prob for target class if binary)."""
        logits = self.model(x)
        probs = F.softmax(logits, dim=-1)
        return probs

    @torch.no_grad()
    def _find_boundary_on_ray(
        self, 
        x_start: torch.Tensor, 
        x_end: torch.Tensor, 
        y_original: int
    ) -> Optional[torch.Tensor]:
        """
        Finds the boundary point on the segment [x_start, x_end] using bisection.
        Assumes x_start is class y_original and x_end is NOT class y_original.
        """
        # Verify endpoints
        start_logits = self.model(x_start)
        start_pred = start_logits.argmax(dim=-1).item()
        
        end_logits = self.model(x_end)
        end_pred = end_logits.argmax(dim=-1).item()
        
        if start_pred != y_original:
            logger.warning("IBS ray search: Start point not original class!")
            return None # Should not happen if logic is correct
            
        if end_pred == y_original:
            # The target sample is also classified as original class (e.g. misclassified or overlapping)
            # We cannot find a boundary on this segment easily if both ends are same class
            return None

        # Precise Bisection / Iterative Refinement
        low = 0.0
        high = 1.0
        
        best_boundary = None
        
        for i in range(self.max_iterations):
            mid_alpha = (low + high) / 2
            x_mid = x_start + mid_alpha * (x_end - x_start)
            
            logits = self.model(x_mid)
            probs = F.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1).item()
            
            # Check for "neutrality" (0.5 prob in binary case, or simply boundary crossing)
            # In multi-class, boundary is where top-2 probs are close?
            # For simplicity, we stick to the class flip definition of boundary.
            
            if pred == y_original:
                low = mid_alpha
            else:
                high = mid_alpha
                best_boundary = x_mid # This is a candidate for "flipped" point (or close to boundary)

            # Convergence check: Difference in alpha or probability?
            # Using alpha precision here as a proxy for precise location
            if (high - low) < 1e-4: # Tolerance
                break
                
        # Final point: average of low and high is the best estimate
        final_alpha = (low + high) / 2
        return x_start + final_alpha * (x_end - x_start)

    def search(
        self, 
        x: np.ndarray, 
        initial_boundary_point: Optional[np.ndarray] = None
    ) -> BoundarySearchResult:
        """
        Run IBS search for input x.
        
        1. Identify target candidates from X_train.
        2. Perform ray search towards candidates.
        3. Select best boundary point (closest to x).
        """
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # 1. Original Prediction
        orig_logits = self.model(x_tensor)
        y_orig = orig_logits.argmax(dim=-1).item()
        
        # 2. Sample Candidates (Target Distribution)
        # Filter X_train for points NOT in y_orig
        mask = (self.y_train != y_orig)
        X_targets = self.X_train[mask]
        
        if len(X_targets) == 0:
            logger.warning("No target class samples found in X_train!")
            return BoundarySearchResult(x, x, None, 0, False, {"error": "no_targets"})
            
        # Randomly select K candidates
        indices = torch.randperm(len(X_targets))[:self.num_candidates]
        target_samples = X_targets[indices].to(self.device).unsqueeze(1) # Add batch dim for consistent shape if needed? 
        # Actually X_targets is (N, D...). target_samples is (K, D...)
        # We need to reshape x_tensor to broadcast or loop.
        
        boundary_candidates = []
        
        # Also include initial_boundary_point if provided (as a direction or endpoint)
        if initial_boundary_point is not None:
            # If we have an FGSM point, it's already on the boundary (roughly).
            # We can try to refine it or treat it as a seed.
            # IBS "refines" it.
            # We can add it to our list of 'directions' or just compare result against it.
            # For now, let's treat it as a fallback or a candidate to beat.
            boundary_candidates.append(torch.tensor(initial_boundary_point, device=self.device).unsqueeze(0))
        
        # 3. Ray Search Loop
        for i in range(len(indices)):
            t_sample = X_targets[indices[i]].unsqueeze(0).to(self.device) # (1, D...)
            
            b_point = self._find_boundary_on_ray(x_tensor, t_sample, y_orig)
            
            if b_point is not None:
                boundary_candidates.append(b_point)
                
        if not boundary_candidates:
            logger.warning("IBS found no boundary points.")
            return BoundarySearchResult(x, x, None, 0, False, {"method": "ibs"})

        # 4. Selection (Distance Minimization)
        best_dist = float('inf')
        best_point = None
        
        for b_cand in boundary_candidates:
            # L2 distance
            dist = torch.norm(b_cand - x_tensor, p=2).item()
            if dist < best_dist:
                best_dist = dist
                best_point = b_cand

        x_boundary_np = best_point.squeeze(0).cpu().numpy()
        
        return BoundarySearchResult(
            x_start=x,
            x_boundary=x_boundary_np,
            x_enemy=x_boundary_np, # Ideally this is the flipped point
            num_steps=self.max_iterations * self.num_candidates, # Approx
            success=True,
            meta={
                "method": "ibs",
                "candidates_checked": len(boundary_candidates),
                "min_dist": best_dist
            }
        )

# Helper function wrapper
def ibs_boundary_search(
    model, 
    x, 
    X_train, 
    y_train, 
    num_candidates=10
) -> Tuple[np.ndarray, bool]:
    searcher = IBSBoundarySearch(model, X_train, y_train, num_candidates=num_candidates)
    res = searcher.search(x)
    return res.x_boundary, res.success

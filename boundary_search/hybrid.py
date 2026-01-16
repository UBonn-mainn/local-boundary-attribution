
import logging
from typing import Optional, Dict, Any, Union, List
import numpy as np
import torch
from .fgsm import FGSMBoundarySearch
from .ibs import IBSBoundarySearch, BoundarySearchResult

logger = logging.getLogger(__name__)

class HybridFGSMIBS:
    """
    Combines FGSM and IBS strategies with a focus on refining the FGSM result.
    
    Strategy:
    1. Run FGSM to find an initial high-quality boundary point.
    2. 'Refine' this solution by exploring directions locally around the FGSM vector
       (and optionally standard IBS global directions towards target samples).
    3. The best point (minimum distance to original) is selected.
    
    This effectively uses FGSM as a 'warm start' for a directional search.
    """
    def __init__(
        self, 
        model: torch.nn.Module,
        X_train: Union[np.ndarray, torch.Tensor],
        y_train: Union[np.ndarray, torch.Tensor],
        device: Optional[torch.device] = None,
        fgsm_params: Optional[Dict[str, Any]] = None,
        ibs_params: Optional[Dict[str, Any]] = None,
    ):
        self.device = device or torch.device("cpu")
        self.model = model
        fgsm_params = fgsm_params or {}
        ibs_params = ibs_params or {}
        
        self.fgsm_searcher = FGSMBoundarySearch(
            model=model, 
            device=self.device, 
            **fgsm_params
        )
        
        # We use the IBS searcher mechanism for ray-casting
        self.ibs_searcher = IBSBoundarySearch(
            model=model, 
            X_train=X_train, 
            y_train=y_train, 
            device=self.device, 
            **ibs_params
        )
        
        # Params for local random refinement
        # If > 0, we generate this many random perturbations of the FGSM direction
        self.num_random_refinements = ibs_params.get("num_candidates", 10)
        self.noise_scale = 0.1 # Std dev of noise relative to vector length?

    def search(self, x: np.ndarray) -> BoundarySearchResult:
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # 1. Run FGSM
        res_fgsm = self.fgsm_searcher.search(x)
        fgsm_pt = None
        
        if res_fgsm.success:
            fgsm_pt = res_fgsm.x_boundary
            logger.debug("Hybrid: FGSM found boundary point with dist %.4f", 
                         np.linalg.norm(fgsm_pt - x))
        else:
            logger.debug("Hybrid: FGSM failed to find boundary.")
            # If FGSM fails, we can fall back to pure IBS or fail.
            # For now, let's treat 'x' or None as start, effectively reducing to pure IBS.

        # 2. Generate Search Candidates (Endpoints for Rays)
        target_endpoints = []
        
        # A) FGSM Direction (The Baseline)
        if fgsm_pt is not None:
             target_endpoints.append(torch.tensor(fgsm_pt, device=self.device).unsqueeze(0))

        # B) Random Perturbations around FGSM ("Local Refinement")
        # The user wants "search directions randomly picked... keeping baseline on decision boundary"
        # We do this by casting rays in random directions and solving for the boundary.
        if fgsm_pt is not None and self.num_random_refinements > 0:
            fgsm_vec = torch.tensor(fgsm_pt - x, device=self.device, dtype=torch.float32)
            for _ in range(self.num_random_refinements):
                # Generate random noise
                noise = torch.randn_like(fgsm_vec) * self.noise_scale * torch.norm(fgsm_vec)
                
                # New candidate direction endpoint
                # We project it out to be somewhat "target-like"
                candidate_endpoint = x_tensor + (fgsm_vec + noise).unsqueeze(0)
                target_endpoints.append(candidate_endpoint)

        # C) Standard IBS Candidates (Data Driven) - Optional
        # If the user strictly wants "randomly picked" (meaning generic), we might skip this.
        # But usually Hybrid implies using data. Let's include a few if we have "budget" left?
        # For now, we strictly follow the new interpretation: FGSM start -> Random Directions.
        # The `ibs_searcher` class naturally picks from X_train, so we can use its logic if we wanted.
        # Let's trust strictly in the "FGSM + Local Perturbation" as the primary "Refined FGSM" strategy.
        
        # 3. Execute Ray Search
        # We reuse IBS internal logic `_find_boundary_on_ray`
        
        orig_logits = self.model(x_tensor)
        y_orig = orig_logits.argmax(dim=-1).item()
        
        boundary_matches = []
        
        # We iterate our manual candidates
        for t_end in target_endpoints:
            # Ray search from x_start to t_end
            b_point = self.ibs_searcher._find_boundary_on_ray(x_tensor, t_end, y_orig)
            if b_point is not None:
                boundary_matches.append(b_point)
                
        # 4. Select Best
        if not boundary_matches:
             # Fallback if perturbations failed (e.g. didn't cross boundary?)
             # If FGSM succeeded, we should have at least that match (unless _find_boundary failed on it)
             if fgsm_pt is not None:
                 return res_fgsm
             return BoundarySearchResult(x, x, None, 0, False, {"method": "hybrid", "error": "no_boundary"})

        best_dist = float('inf')
        best_point = None
        
        for b_cand in boundary_matches:
            dist = torch.norm(b_cand - x_tensor, p=2).item()
            if dist < best_dist:
                best_dist = dist
                best_point = b_cand
                
        x_boundary_np = best_point.squeeze(0).cpu().numpy()
        
        return BoundarySearchResult(
            x_start=x,
            x_boundary=x_boundary_np,
            x_enemy=x_boundary_np,
            num_steps=self.num_random_refinements, # Proxy
            success=True,
            meta={
                "method": "hybrid_fgsm_perturbed",
                "fgsm_success": res_fgsm.success if res_fgsm else False,
                "candidates_checked": len(boundary_matches),
                "min_dist": best_dist
            }
        )

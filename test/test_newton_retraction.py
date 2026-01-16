
import sys
import os
import torch
import torch.nn as nn
import logging
import numpy as np

# Add parent directory to path to import boundary_search
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from boundary_search.boundary_walker import BoundaryCrawler

# Configure logging to see the debug prints from the crawler
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

class ToyModel(nn.Module):
    """
    Simple model: Decision boundary is a sphere of radius 5.0 centered at 0.
    Class 0: Inside sphere (norm < 5)
    Class 1: Outside sphere (norm > 5)
    """
    def __init__(self):
        super().__init__()
        self.radius = 5.0
        
    def forward(self, x):
        # x shape: [B, D]
        norm = torch.norm(x, dim=1, keepdim=True)
        # Logits: [diff, 0] such that if diff > 0 (norm < 5), Class 0 wins.
        # We want Logit0 > Logit1 when norm < 5.
        
        # valid: norm < 5 => val = 5 - norm > 0.
        # Logit0 = val, Logit1 = 0.
        val = self.radius - norm
        
        logit0 = val
        logit1 = torch.zeros_like(val)
        
        return torch.cat([logit0, logit1], dim=1)

def test_newton_convergence():
    print("--- Starting Newton Retraction Test ---")
    
    device = torch.device("cpu")
    model = ToyModel()
    
    # Start point: Inside sphere, near center
    x_start = np.array([1.0, 1.0, 1.0], dtype=np.float32) # Norm ~ 1.73
    
    # Init crawler
    # Mode gradient is required for Newton path
    crawler = BoundaryCrawler(
        model=model, 
        device=device,
        crawl_params={
            "mode": "gradient",
            "max_iterations": 5,
            "step_size": 0.5,
            "bisection_steps": 5 # Low bisection steps to rely on Newton
        }
    )
    
    # Run search
    print(f"Start Point: {x_start}")
    result = crawler.search(x_start)
    
    print(f"Search Success: {result.success}")
    print(f"Total Steps: {result.num_steps}")
    print(f"Final Boundary Point: {result.x_boundary}")
    
    final_norm = np.linalg.norm(result.x_boundary)
    print(f"Final Norm (Target 5.0): {final_norm:.4f}")
    
    # Verify
    if abs(final_norm - 5.0) < 0.1:
        print("[PASS] Final point is on the decision boundary.")
    else:
        print("[FAIL] Final point is NOT on the decision boundary.")
        
    # Check if we improved the distance
    # Original distance to boundary from (1,1,1) is 5 - sqrt(3) ~ 3.26
    # Ideally we just hit the boundary.
    # The 'closest' point on boundary to (1,1,1) is 5/sqrt(3)*(1,1,1) = (2.88, 2.88, 2.88).
    # Distance is |(2.88,3) - (1,1,1)| = 3.26.
    
    dist = np.linalg.norm(result.x_boundary - x_start)
    print(f"Distance found: {dist:.4f} (Ideal: {5.0 - np.linalg.norm(x_start):.4f})")

if __name__ == "__main__":
    test_newton_convergence()


import logging
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

from boundary_search.fgsm import fgsm_boundary_search
from boundary_search.ibs import ibs_boundary_search, IBSBoundarySearch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_simple_dataset():
    # 2 blobs
    # Class 0 centered at (-2, 0)
    # Class 1 centered at (2, 0)
    np.random.seed(42)
    n = 100
    X0 = np.random.randn(n, 2) * 0.5 + np.array([-2, 0])
    y0 = np.zeros(n)
    X1 = np.random.randn(n, 2) * 0.5 + np.array([2, 0])
    y1 = np.ones(n)
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    return X, y

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def train_model(X, y):
    model = SimpleMLP()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    
    for i in range(200):
        logits = model(Xt)
        loss = torch.nn.functional.cross_entropy(logits, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model

def main():
    device = torch.device('cpu') # Keep simple for now
    X, y = get_simple_dataset()
    model = train_model(X, y)
    model.eval()
    
    # Select a test point deep in Class 0
    x_test_np = np.array([-3.0, 0.0], dtype=np.float32)
    x_test = torch.tensor(x_test_np).unsqueeze(0)
    
    print(f"--- Comparison: Input {x_test_np} (Class 0) ---")
    
    # 1. FGSM Only
    t0 = time.time()
    fgsm_pt, fgsm_succ = fgsm_boundary_search(model, x_test)
    t_fgsm = time.time() - t0
    fgsm_dist = torch.norm(fgsm_pt - x_test).item()
    print(f"[FGSM] Success: {fgsm_succ}, Dist: {fgsm_dist:.4f}, Time: {t_fgsm:.4f}s")
    
    # 2. IBS Only (Pure)
    # Search towards 10 random targets from Class 1
    t0 = time.time()
    ibs_searcher = IBSBoundarySearch(model, X, y, num_candidates=10, device=device)
    res_ibs = ibs_searcher.search(x_test_np)
    t_ibs = time.time() - t0
    
    ibs_pt = torch.tensor(res_ibs.x_boundary)
    ibs_dist = torch.norm(ibs_pt - x_test).item()
    print(f"[IBS-Only] Success: {res_ibs.success}, Dist: {ibs_dist:.4f}, Time: {t_ibs:.4f}s")
    
    # 3. FGSM + IBS Refinement
    # Use FGSM point as a 'seed' or candidate
    # NOTE: Our current IBS implementation takes `initial_boundary_point` which is added to the candidate list.
    # So we effectively check K random targets + 1 FGSM target.
    t0 = time.time()
    
    # Step A: FGSM
    curr_fgsm_pt, curr_fgsm_succ = fgsm_boundary_search(model, x_test)
    
    # Step B: IBS Refinement
    # We pass the FGSM point. IBS will try to perform exact search along the ray x_test -> fgsm_pt
    res_refined = ibs_searcher.search(x_test_np, initial_boundary_point=curr_fgsm_pt.cpu().numpy())
    t_refined = time.time() - t0
    
    refined_pt = torch.tensor(res_refined.x_boundary)
    refined_dist = torch.norm(refined_pt - x_test).item()
    print(f"[FGSM+IBS] Success: {res_refined.success}, Dist: {refined_dist:.4f}, Time: {t_refined:.4f}s")
    
    # Summary
    print("\n--- Summary ---")
    print(f"IBS-Only Improvement over FGSM (Dist): {fgsm_dist - ibs_dist:.4f}")
    print(f"FGSM+IBS Improvement over FGSM (Dist): {fgsm_dist - refined_dist:.4f}")
    print(f"FGSM+IBS vs IBS-Only (Dist): {ibs_dist - refined_dist:.4f} (Positive means Refined is better)")

if __name__ == "__main__":
    main()

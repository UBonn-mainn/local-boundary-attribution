
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path to import boundary_search
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from boundary_search.boundary_walker import BoundaryCrawler
from utils.models.train_linear_model import train_model_memory as train_linear_model
from utils.models.train_mlp_model import train_mlp_model_memory as train_mlp_model
from utils.data.load_model import LinearClassifier

import argparse

def test_8d_boundary_search(data_path: str, model_type: str = "linear"):
    # 1. Load Data
    csv_path = Path(data_path)
    
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return

    print(f"Loading data from {csv_path}")
    
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(int)
    
    print(f"Data Shape: {X.shape}")
    
    # 2. Train Model using shared utility
    print(f"Training {model_type} Model...")
    
    if model_type == "linear":
        model = train_linear_model(
            X_train=X, 
            y_train=y,
            input_dim=X.shape[1],
            epochs=10,
            batch_size=32
        )
    else:
        # MLP
         model = train_mlp_model(
            X_train=X, 
            y_train=y,
            input_dim=X.shape[1],
            epochs=20,
            batch_size=32
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Test running on: {device}")
    model.to(device)
    model.eval()

    # 3. Pick a sample
    # Pick a sample that is clearly in one class
    idx = 0
    x_test = X[idx]
    y_test = y[idx]
    
    print(f"Testing on Sample {idx} (Class {y_test})")
    print(f"Sample Vector: {x_test}")
    
    # 4. Run Crawler
    crawler = BoundaryCrawler(
        model=model,
        device=device,
        crawl_params={
            "mode": "gradient",
            "max_iterations": 20,
            "step_size": 0.1
        },
        fgsm_params={
            "step_size": 0.1,
            "max_steps": 50
        }
    )
    
    result = crawler.search(x_test)
    
    print("--- Crawler Result ---")
    print(f"Success: {result.success}")
    print(f"Steps: {result.num_steps}")
    print(f"Final Dist: {np.linalg.norm(result.x_boundary - x_test):.4f}")
    
    # Verify the point is on boundary (logits close)
    x_bound_t = torch.tensor(result.x_boundary, dtype=torch.float32).unsqueeze(0)
    logits = model(x_bound_t)
    probs = torch.softmax(logits, dim=1)
    print(f"Final Probs: {probs.detach().numpy()}")
    
    if abs(probs[0, 0].item() - 0.5) < 0.1:
        print("[PASS] Point is on decision boundary.")
    else:
        print("[FAIL] Point is NOT on decision boundary (probs not ~0.5).")

    # --- 5. Visualization (PCA projection) ---
    print("\nGenerating Visualization...")
    import matplotlib.pyplot as plt
    
    # 1. Simple PCA implementation using SVD to avoid sklean dependency
    # Center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:2] # Top 2 principal components (2, 8)
    
    # Project Data
    X_pca = X_centered @ components.T
    
    # Project Points of Interest
    # We need to center them using the TRAINING set mean
    start_centered = x_test - X_mean
    bound_centered = result.x_boundary - X_mean
    
    start_pca = start_centered @ components.T
    bound_pca = bound_centered @ components.T
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Scatter all points
    plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], alpha=0.3, label='Class 0', c='blue')
    plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], alpha=0.3, label='Class 1', c='red')
    
    # Plot Trajectory (Start -> Boundary)
    plt.plot([start_pca[0], bound_pca[0]], [start_pca[1], bound_pca[1]], 
             'k--', linewidth=2, label='Crawling Path')
    
    # Mark Start and End
    plt.scatter(start_pca[0], start_pca[1], c='green', s=150, marker='*', label='Start (Original)', edgecolors='k', zorder=5)
    plt.scatter(bound_pca[0], bound_pca[1], c='yellow', s=150, marker='X', label='Found Boundary', edgecolors='k', zorder=5)
    
    plt.title(f"8D Boundary Search Visualization (PCA Projection)\nDistance: {result.meta['final_dist']:.4f} | Steps: {result.num_steps}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_file = Path("test/crawling_path_8d.png")
    plt.savefig(out_file)
    print(f"Visualization saved to {out_file.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Boundary Search on 8D Data")
    parser.add_argument("--data_path", type=str, default="data/linearly_separable_data_8D.csv", help="Path to CSV data")
    parser.add_argument("--model_type", type=str, default="linear", choices=["linear", "mlp"], help="Model type (linear or mlp)")
    
    args = parser.parse_args()
    
    test_8d_boundary_search(args.data_path, args.model_type)

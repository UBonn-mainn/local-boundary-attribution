
import sys
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path to import boundary_search
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from boundary_search.boundary_walker import BoundaryCrawler
from utils.models.train_linear_model import train_model_memory as train_linear_model
from utils.models.train_mlp_model import train_mlp_model_memory as train_mlp_model
from utils.data.load_model import load_model

import argparse


def test_boundary_search(
    data_path: str, 
    model_path: str = None, 
    model_type: str = "linear",
    sample_idx: int = 0,
):
    """
    Test the BoundaryCrawler on arbitrary tabular data.
    
    Supports two modes:
        - Provide --model_path to load a pre-trained model (.pth checkpoint)
        - Omit --model_path to train a fresh model on the data
    
    The data CSV should have features in all columns except the last,
    which is treated as the integer class label.
    """
    # 1. Load Data
    csv_path = Path(data_path)
    
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return

    print(f"Loading data from {csv_path}")
    
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(int)
    
    input_dim = X.shape[1]
    num_classes = len(np.unique(y))
    
    print(f"Data Shape: {X.shape}, Classes: {num_classes}, Input Dim: {input_dim}")
    
    # 2. Load or train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if model_path is not None:
        print(f"Loading model from {model_path}")
        model = load_model(
            model_path, 
            input_dim=input_dim, 
            model_type=model_type, 
            num_classes=num_classes
        )
    else:
        print(f"Training {model_type} model on data...")
        if model_type == "linear":
            model = train_linear_model(
                X_train=X, 
                y_train=y,
                input_dim=input_dim,
                epochs=10,
                batch_size=32
            )
        else:
            model = train_mlp_model(
                X_train=X, 
                y_train=y,
                input_dim=input_dim,
                epochs=20,
                batch_size=32
            )
    
    model.to(device)
    model.eval()

    # 3. Pick a sample
    if sample_idx >= len(X):
        print(f"Error: sample_idx {sample_idx} out of range (dataset has {len(X)} samples)")
        return
        
    x_test = X[sample_idx]
    y_test = y[sample_idx]
    
    print(f"\nTesting on Sample {sample_idx} (Class {y_test})")
    print(f"Sample Vector: {x_test}")
    
    # 4. Run Crawler
    crawler = BoundaryCrawler(
        model=model,
        device=device,
        crawl_params={
            "max_iterations": 20,
            "step_size": 0.1,
            "max_target_classes": min(5, num_classes - 1),
        },
        fgsm_params={
            "step_size": 0.1,
            "max_steps": 50
        }
    )
    
    result = crawler.search(x_test)
    
    print("\n--- Crawler Result ---")
    print(f"Success: {result.success}")
    print(f"Steps: {result.num_steps}")
    print(f"Final Dist: {np.linalg.norm(result.x_boundary - x_test):.4f}")
    
    # Verify the point is on boundary (top-2 probs should be close)
    x_bound_t = torch.tensor(result.x_boundary, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(x_bound_t)
        probs = torch.softmax(logits, dim=1)
        
    top2_probs, top2_classes = torch.topk(probs[0], min(2, num_classes))
    print(f"All Probs: {probs.cpu().numpy()}")
    print(f"Top-2 Classes: {top2_classes.cpu().tolist()}, Probs: {top2_probs.cpu().tolist()}")
    
    # Boundary check: top-2 probabilities should be close to each other
    if num_classes >= 2:
        prob_gap = abs(top2_probs[0].item() - top2_probs[1].item())
        if prob_gap < 0.2:
            print(f"[PASS] Point is on decision boundary (top-2 prob gap = {prob_gap:.4f})")
        else:
            print(f"[FAIL] Point is NOT on decision boundary (top-2 prob gap = {prob_gap:.4f})")

    # --- 5. Visualization (PCA projection) ---
    print("\nGenerating Visualization...")
    import matplotlib.pyplot as plt
    
    # PCA via SVD
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:2]
    
    X_pca = X_centered @ components.T
    
    start_pca = (x_test - X_mean) @ components.T
    bound_pca = (result.x_boundary - X_mean) @ components.T
    
    plt.figure(figsize=(10, 8))
    
    # Color by class (works for any number of classes)
    cmap = plt.cm.get_cmap("tab10", num_classes)
    for c in range(num_classes):
        mask = y == c
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.3, label=f'Class {c}', c=[cmap(c)])
    
    # Plot trajectory
    plt.plot([start_pca[0], bound_pca[0]], [start_pca[1], bound_pca[1]], 
             'k--', linewidth=2, label='Crawling Path')
    
    plt.scatter(start_pca[0], start_pca[1], c='green', s=150, marker='*', 
                label='Start (Original)', edgecolors='k', zorder=5)
    plt.scatter(bound_pca[0], bound_pca[1], c='yellow', s=150, marker='X', 
                label='Found Boundary', edgecolors='k', zorder=5)
    
    final_dist = result.meta.get('final_dist', np.linalg.norm(result.x_boundary - x_test))
    plt.title(f"Boundary Search (PCA Projection) | {input_dim}D data, {num_classes} classes\n"
              f"Distance: {final_dist:.4f} | Steps: {result.num_steps}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_file = Path("test/crawling_path.png")
    plt.savefig(out_file)
    print(f"Visualization saved to {out_file.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Boundary Search on arbitrary tabular data")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to CSV data (features + last column = label)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to pre-trained model (.pth). If omitted, trains a fresh model.")
    parser.add_argument("--model_type", type=str, default="linear", choices=["linear", "mlp"],
                        help="Model architecture type")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Index of the sample to test on")
    
    args = parser.parse_args()
    
    test_boundary_search(
        data_path=args.data_path, 
        model_path=args.model_path,
        model_type=args.model_type,
        sample_idx=args.sample_idx,
    )

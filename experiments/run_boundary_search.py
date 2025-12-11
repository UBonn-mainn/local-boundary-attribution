
import argparse
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from models.load_model import load_model
from boundary_search.fgsm import fgsm_boundary_search
from evaluation.visualize_decision_boundary import plot_decision_boundary

def main():
    parser = argparse.ArgumentParser(description="Run boundary search on a dataset.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "linear"], help="Type of model")
    parser.add_argument("--save_dir", type=str, default="results/boundary_search", help="Directory to save results")
    parser.add_argument("--step_size", type=float, default=0.01, help="FGSM step size")
    parser.add_argument("--max_iters", type=int, default=50, help="Max iterations")
    
    args = parser.parse_args()
    
    # Setup paths
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    print(f"Loading data from {args.data_path}...")
    data = np.loadtxt(args.data_path, delimiter=",", skiprows=1)
    X = data[:, :-1].astype(np.float32)
    y = data[:, -1].astype(np.float32)
    
    # Load Model
    print(f"Loading model from {args.model_path} ({args.model_type})...")
    model = load_model(args.model_path, input_dim=X.shape[1], model_type=args.model_type)
    
    # Run Search
    print("Running boundary search...")
    results = []
    boundary_points = []
    
    for i in tqdm(range(len(X))):
        x_input = torch.tensor(X[i], dtype=torch.float32)
        
        # Run FGSM
        # Note: clamp is None for this synthetic data, but typically (0,1) for images
        b_point, success = fgsm_boundary_search(
            model, 
            x_input, 
            step_size=args.step_size, 
            max_iters=args.max_iters, 
            clamp=None, 
            refine_steps=10
        )
        
        b_point_np = b_point.numpy()
        dist = np.linalg.norm(b_point_np - X[i])
        
        results.append({
            "original_index": i,
            "original_class": y[i],
            "success": success,
            "boundary_x1": b_point_np[0],
            "boundary_x2": b_point_np[1],
            "distance": dist
        })
        
        if success:
            boundary_points.append(b_point_np)
            
    # Save Results
    df_results = pd.DataFrame(results)
    csv_path = save_dir / "boundary_points.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    
    print(f"Success Rate: {df_results['success'].mean() * 100:.2f}%")
    print(f"Mean Distance: {df_results[df_results['success']]['distance'].mean():.4f}")
    
    # Visualize
    if boundary_points:
        print("Generating visualization...")
        bp_array = np.array(boundary_points)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S") # Need datetime import or just overwrite
        plot_path = save_dir / f"boundary_vis_{args.model_type}.png"
        
        plot_decision_boundary(
            model, 
            X, 
            y, 
            output_path=str(plot_path), 
            extra_scatter=(bp_array, "Boundary Points")
        )

if __name__ == "__main__":
    from datetime import datetime # Late import to avoid moving top-level imports
    main()

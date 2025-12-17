
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import Optional

def plot_decision_boundary(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    output_path: Optional[str] = None,
    resolution: int = 200,
    device: str = "cpu",
    extra_scatter = None
):
    """
    Plots the decision boundary of a PyTorch model for 2D data.
    
    Args:
        model: Trained PyTorch model.
        X: Input data (N, 2).
        y: Labels (N,).
        output_path: If provided, saves the plot to this path.
        resolution: Grid resolution for the background mesh.
        device: Device to run inference on.
        extra_scatter: Optional tuple (points, label) to overlay extra points (e.g., boundary points).
    """
    if X.shape[1] != 2:
        raise ValueError("Data must be 2D for visualization.")
        
    model.eval()
    model.to(device)
    
    # Define bounds
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    # Flatten and predict
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = model(grid_tensor)
        preds = logits.argmax(dim=1).cpu().numpy()
        
    Z = preds.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    
    # Plot contour
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=50, alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Classes")
    
    # Plot extra points
    if extra_scatter is not None:
        points, label = extra_scatter
        plt.scatter(points[:, 0], points[:, 1], c='green', marker='x', s=100, linewidths=2, label=label)
        plt.legend()
    
    plt.title("Model Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to {output_path}")
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    from utils.data.load_model import load_model
    from utils.data.dataset_utils import load_dataset_from_csv

    parser = argparse.ArgumentParser(description="Visualize model decision boundary.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "linear"], help="Type of model to load")
    parser.add_argument("--save_path", type=str, default="figures/decision_boundary.png", help="Path to save the plot")
    
    args = parser.parse_args()
    
    # Load Data
    print(f"Loading data from {args.data_path}...")
    X, y = load_dataset_from_csv(args.data_path)
    
    # Load Model
    print(f"Loading model from {args.model_path} (Type: {args.model_type})...")
    model = load_model(args.model_path, input_dim=X.shape[1], model_type=args.model_type)
    
    # Visualize
    print("Generating plot...")
    plot_decision_boundary(model, X, y, output_path=args.save_path)

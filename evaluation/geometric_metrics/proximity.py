import torch

def l2_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute L2 distance between flattened tensors."""
    return torch.norm(x.view(-1) - y.view(-1), p=2).item()

def proximity_score(x, boundary_x):
    """
    Returns the L2 distance between sample and its boundary point.
    Lower = closer to decision boundary = better baseline.
    """
    return l2_distance(x, boundary_x)

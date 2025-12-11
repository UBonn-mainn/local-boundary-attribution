import torch

def l2_disagreement(a, b):
    """Euclidean distance between two attribution vectors."""
    return torch.norm(a.flatten() - b.flatten(), p=2).item()

def sign_disagreement(a, b):
    """Percentage of features with differing signs."""
    s1 = torch.sign(a.flatten())
    s2 = torch.sign(b.flatten())
    return (s1 != s2).float().mean().item()

def topk_overlap(a, b, k=10):
    """
    Computes overlap of top-k important features.
    Higher = more agreement.
    """
    a_top = torch.topk(a.abs().flatten(), k).indices
    b_top = torch.topk(b.abs().flatten(), k).indices
    return len(set(a_top.tolist()) & set(b_top.tolist())) / k

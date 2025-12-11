import torch

def baseline_shift(a_baseline, a_new_baseline):
    """
    Compare attribution vectors produced with two different baselines.
    """
    delta = torch.norm(a_baseline.flatten() - a_new_baseline.flatten(), p=1).item()
    return delta

def relative_baseline_change(a_old, a_new):
    """
    Normalized difference measure (L2).
    """
    return torch.norm(a_old - a_new).item() / (torch.norm(a_old).item() + 1e-9)

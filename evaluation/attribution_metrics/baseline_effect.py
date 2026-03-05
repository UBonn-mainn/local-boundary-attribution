import torch


def attribution_delta_norm(
    attr_a: torch.Tensor,
    attr_b: torch.Tensor,
    p: int = 2,
) -> float:
    """
    Norm of difference between two attributions (e.g., mean-baseline vs boundary-baseline).
    """
    da = (attr_a - attr_b).detach().cpu().flatten().float()
    return float(torch.norm(da, p=p))


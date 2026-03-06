import torch

import torch.nn.functional as F

from utils.common.evaluation import to_2d

def boundary_alignment_cosine(
    attr: torch.Tensor,
    x: torch.Tensor,
    x_boundary: torch.Tensor,
) -> float:
    """
    cos( attr, (x_boundary - x) ) — does attribution align with the boundary direction?
    """
    x = to_2d(x).detach().cpu()
    xb = to_2d(x_boundary).detach().cpu()
    a = to_2d(attr).detach().cpu()

    direction = (xb - x).flatten().float()
    a_vec = a.flatten().float()
    return float(F.cosine_similarity(a_vec, direction, dim=0).item())



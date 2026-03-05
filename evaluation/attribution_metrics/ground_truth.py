from typing import Optional

import torch
from torch import nn

from utils.common.evaluation import get_target_from_logits, to_2d, select_logit


def true_gradient_attribution(
    model: nn.Module,
    x: torch.Tensor,
    target: Optional[int] = None,
) -> torch.Tensor:
    """
    Computes |∇_x f_t(x)| as a 'numerical ground truth' for synthetic settings.
    Returns tensor same shape as x (tabular).
    """
    model.eval()
    x = to_2d(x)
    device = next(model.parameters()).device
    x = x.to(device).detach().requires_grad_(True)

    logits = model(x)
    tgt = get_target_from_logits(logits) if target is None else torch.full(
        (x.size(0),), int(target), device=device, dtype=torch.long
    )

    score = select_logit(logits, tgt).sum()
    score.backward()

    grad = x.grad.detach()
    return grad.abs()


def cosine_with_true_gradient(
    attr: torch.Tensor,
    true_grad_attr: torch.Tensor,
) -> float:
    """
    Cosine similarity between attribution and 'true' gradient attribution.
    """
    a = attr.detach().cpu().flatten().float()
    g = true_grad_attr.detach().cpu().flatten().float()
    return float(F.cosine_similarity(a, g, dim=0).item())

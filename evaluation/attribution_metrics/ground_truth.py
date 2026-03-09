from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from utils.common.evaluation import get_target_from_logits, to_2d, select_logit


def true_gradient_attribution(
    model: nn.Module,
    x: torch.Tensor,
    target: Optional[int] = None,
) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device

    # Detect whether model is image-based
    has_conv2d = any(isinstance(m, nn.Conv2d) for m in model.modules())

    if has_conv2d:
        # Image input handling
        if x.ndim == 2:
            # H x W  -> 1 x 1 x H x W
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            # C x H x W  -> 1 x C x H x W
            x = x.unsqueeze(0)
        elif x.ndim != 4:
            raise ValueError(f"Unsupported image input shape: {tuple(x.shape)}")
    else:
        # Tabular input handling
        x = to_2d(x)

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
    a = attr.detach().cpu().flatten().float()
    g = true_grad_attr.detach().cpu().flatten().float()
    return float(F.cosine_similarity(a, g, dim=0).item())
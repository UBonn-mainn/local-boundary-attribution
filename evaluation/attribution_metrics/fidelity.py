from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn

from utils.common.evaluation import (
    to_2d,
    get_target_from_logits,
    select_logit,
    rank_indices_by_attr,
    auc_trapz,
    replace_features,
)


@dataclass
class FidelityResult:
    deletion_auc: float
    insertion_auc: float
    deletion_delta_topk: float
    insertion_delta_topk: float


@torch.no_grad()
def fidelity_deletion_insertion(
    model: nn.Module,
    x: torch.Tensor,
    attr: torch.Tensor,
    baseline: torch.Tensor,
    target: Optional[int] = None,
    steps: int = 25,
    topk: int = 10,
    use_logits: bool = True,
) -> FidelityResult:
    """
    Works for both tabular data and image tensors.
    For images, attribution / replacement is done in flattened space,
    but model evaluation uses the original image shape.
    """
    model.eval()
    device = next(model.parameters()).device

    # --------------------------------------------------
    # Ensure batch dimension and remember original shape
    # --------------------------------------------------
    has_conv2d = any(isinstance(m, nn.Conv2d) for m in model.modules())

    if has_conv2d:
        # image case
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # H,W -> 1,1,H,W
        elif x.ndim == 3:
            x = x.unsqueeze(0)  # C,H,W -> 1,C,H,W

        if baseline.ndim == 2:
            baseline = baseline.unsqueeze(0).unsqueeze(0)
        elif baseline.ndim == 3:
            baseline = baseline.unsqueeze(0)

        if attr.ndim == 2:
            attr = attr.unsqueeze(0).unsqueeze(0)
        elif attr.ndim == 3:
            attr = attr.unsqueeze(0)
    else:
        # tabular case
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if baseline.ndim == 1:
            baseline = baseline.unsqueeze(0)
        if attr.ndim == 1:
            attr = attr.unsqueeze(0)

    x = x.to(device)
    baseline = baseline.to(device)
    attr = attr.to(device)

    original_shape = x.shape

    # --------------------------------------------------
    # Flatten only for feature ranking / replacement
    # --------------------------------------------------
    x_flat = x.view(x.size(0), -1)
    baseline_flat = baseline.view(baseline.size(0), -1)
    attr_flat = attr.view(attr.size(0), -1)

    D = x_flat.size(1)
    steps = max(2, int(steps))

    def reshape_for_model(z_flat: torch.Tensor) -> torch.Tensor:
        return z_flat.view(original_shape)

    def model_score(z_flat: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        z = reshape_for_model(z_flat)
        out = model(z)
        score = select_logit(out, tgt)
        return score if use_logits else torch.sigmoid(score)

    # --------------------------------------------------
    # Initial target
    # --------------------------------------------------
    logits = model(x)
    tgt = get_target_from_logits(logits) if target is None else (
        torch.full((x.size(0),), int(target), device=device, dtype=torch.long)
        if isinstance(target, int)
        else target.to(device)
    )

    # --------------------------------------------------
    # Importance ranking
    # --------------------------------------------------
    order = rank_indices_by_attr(attr_flat, descending=True)  # (B, D)
    ks = torch.linspace(0, D, steps=steps, device=device).long()

    del_scores: List[torch.Tensor] = []
    ins_scores: List[torch.Tensor] = []

    # --------------------------------------------------
    # Deletion curve
    # --------------------------------------------------
    for k in ks:
        if k.item() == 0:
            xk_flat = x_flat
        else:
            idx = order[:, :k]
            xk_flat = replace_features(x_flat, baseline_flat, idx)

        score = model_score(xk_flat, tgt)
        del_scores.append(score.detach().cpu())

    # --------------------------------------------------
    # Insertion curve
    # --------------------------------------------------
    for k in ks:
        if k.item() == 0:
            xk_flat = baseline_flat
        else:
            idx = order[:, :k]
            xk_flat = replace_features(baseline_flat, x_flat, idx)

        score = model_score(xk_flat, tgt)
        ins_scores.append(score.detach().cpu())

    # --------------------------------------------------
    # Curves + AUC
    # --------------------------------------------------
    del_curve = torch.stack([s.mean() for s in del_scores])
    ins_curve = torch.stack([s.mean() for s in ins_scores])

    x_axis = ks.detach().cpu().float() / float(D)
    deletion_auc = auc_trapz(del_curve, x_axis)
    insertion_auc = auc_trapz(ins_curve, x_axis)

    # --------------------------------------------------
    # Top-k deltas
    # --------------------------------------------------
    k = min(int(topk), D)
    idxk = order[:, :k]

    x_del_flat = replace_features(x_flat, baseline_flat, idxk)
    x_ins_flat = replace_features(baseline_flat, x_flat, idxk)

    base_score = model_score(x_flat, tgt).mean()
    del_score = model_score(x_del_flat, tgt).mean()
    ins_score = model_score(x_ins_flat, tgt).mean()
    baseline_score = model_score(baseline_flat, tgt).mean()

    deletion_delta_topk = float((base_score - del_score).abs().detach().cpu())
    insertion_delta_topk = float((ins_score - baseline_score).abs().detach().cpu())

    return FidelityResult(
        deletion_auc=deletion_auc,
        insertion_auc=insertion_auc,
        deletion_delta_topk=deletion_delta_topk,
        insertion_delta_topk=insertion_delta_topk,
    )
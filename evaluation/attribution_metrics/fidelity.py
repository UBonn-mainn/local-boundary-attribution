from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.common.evaluation import to_2d, get_target_from_logits, select_logit, spearmanr_torch, rank_indices_by_attr, \
    auc_trapz, replace_features


# -----------------------------
# Fidelity (Deletion/Insertion)
# -----------------------------

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
    Deletion: start from x, progressively replace most important features with baseline.
    Insertion: start from baseline, progressively insert most important features from x.

    We track selected-class score (logit by default) across steps and compute AUC.
    For deletion, lower curve => "good" (score drops quickly). We still report AUC.
    For insertion, higher curve => "good".
    """
    model.eval()
    x = to_2d(x)
    baseline = to_2d(baseline)
    attr = to_2d(attr)

    device = next(model.parameters()).device
    x = x.to(device)
    baseline = baseline.to(device)
    attr = attr.to(device)

    logits = model(x)
    tgt = get_target_from_logits(logits) if target is None else (
        torch.full((x.size(0),), int(target), device=device, dtype=torch.long)
        if isinstance(target, int) else target.to(device)
    )

    # importance order
    order = rank_indices_by_attr(attr, descending=True)  # (B,D)

    # step sizes = how many features changed per step
    D = x.size(1)
    steps = max(2, int(steps))
    ks = torch.linspace(0, D, steps=steps).long()  # number of features replaced/inserted

    del_scores: List[torch.Tensor] = []
    ins_scores: List[torch.Tensor] = []

    # Deletion curve
    for k in ks:
        if k.item() == 0:
            xk = x
        else:
            idx = order[:, :k]
            xk = replace_features(x, baseline, idx)
        out = model(xk)
        score = select_logit(out, tgt) if use_logits else torch.sigmoid(select_logit(out, tgt))
        del_scores.append(score.detach().cpu())

    # Insertion curve
    for k in ks:
        if k.item() == 0:
            xk = baseline
        else:
            idx = order[:, :k]
            xk = replace_features(baseline, x, idx)  # insert from x into baseline
        out = model(xk)
        score = select_logit(out, tgt) if use_logits else torch.sigmoid(select_logit(out, tgt))
        ins_scores.append(score.detach().cpu())

    # average across batch
    del_curve = torch.stack([s.mean() for s in del_scores])  # (steps,)
    ins_curve = torch.stack([s.mean() for s in ins_scores])  # (steps,)

    x_axis = ks.float() / float(D)
    deletion_auc = auc_trapz(del_curve, x_axis)
    insertion_auc = auc_trapz(ins_curve, x_axis)

    # top-k deltas (single-step)
    k = min(int(topk), D)
    idxk = order[:, :k]
    x_del = replace_features(x, baseline, idxk)
    x_ins = replace_features(baseline, x, idxk)

    base_score = select_logit(model(x), tgt).mean()
    del_score = select_logit(model(x_del), tgt).mean()
    ins_score = select_logit(model(x_ins), tgt).mean()

    deletion_delta_topk = float((base_score - del_score).abs().detach().cpu())
    insertion_delta_topk = float((ins_score - select_logit(model(baseline), tgt).mean()).abs().detach().cpu())

    return FidelityResult(
        deletion_auc=deletion_auc,
        insertion_auc=insertion_auc,
        deletion_delta_topk=deletion_delta_topk,
        insertion_delta_topk=insertion_delta_topk,
    )



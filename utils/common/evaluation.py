from typing import Optional, Any

import torch


def to_2d(x: torch.Tensor) -> torch.Tensor:
    """Ensure shape (B, D) for tabular inputs."""
    if x.ndim == 1:
        return x.unsqueeze(0)
    return x


def safe_tensor(x, device=None, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.tensor(x)
    if device is not None:
        t = t.to(device)
    return t.to(dtype)


def get_target_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Infer predicted class index from logits (B,C) or (B,) binary logit."""
    if logits.ndim == 1:
        return (logits > 0).long()
    return logits.argmax(dim=-1)


def select_logit(logits: torch.Tensor, target: Any) -> torch.Tensor:
    """
    Returns selected logit for each sample:
    - logits: (B,C) -> (B,)
    - logits: (B,) -> (B,)
    """
    if logits.ndim == 1:
        return logits
    if isinstance(target, int):
        return logits[:, target]
    # target is tensor of shape (B,)
    return logits.gather(1, target.view(-1, 1)).squeeze(1)


def rank_indices_by_attr(attr: torch.Tensor, descending: bool = True) -> torch.Tensor:
    """Returns feature indices sorted by |attr| descending by default."""
    a = attr.abs()
    return torch.argsort(a, dim=-1, descending=descending)


def replace_features(
    x: torch.Tensor,
    baseline: torch.Tensor,
    feat_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Replace selected features in x with baseline values.
    x, baseline: (B,D)
    feat_idx: (B,k) or (k,)
    """
    x = x.clone()
    baseline = baseline.expand_as(x)

    if feat_idx.ndim == 1:
        # same indices for all in batch
        x[:, feat_idx] = baseline[:, feat_idx]
    else:
        # per-sample indices
        for b in range(x.size(0)):
            x[b, feat_idx[b]] = baseline[b, feat_idx[b]]
    return x


def auc_trapz(y: torch.Tensor, x: Optional[torch.Tensor] = None) -> float:
    """
    AUC with trapezoid rule.
    y: (T,)
    x: (T,) optional; defaults to uniform spacing.
    """
    y = y.detach().cpu().float()
    if x is None:
        x = torch.linspace(0, 1, steps=y.numel())
    else:
        x = x.detach().cpu().float()
    return float(torch.trapz(y, x))


def spearmanr_torch(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Spearman rank correlation without scipy.
    a,b: (D,)
    """
    a = a.detach().cpu().float().flatten()
    b = b.detach().cpu().float().flatten()

    # ranks with average for ties (simple approx: argsort twice; ties rare in float attributions)
    ra = torch.argsort(torch.argsort(a))
    rb = torch.argsort(torch.argsort(b))

    ra = ra.float()
    rb = rb.float()

    ra = (ra - ra.mean()) / (ra.std(unbiased=False) + 1e-12)
    rb = (rb - rb.mean()) / (rb.std(unbiased=False) + 1e-12)
    return float((ra * rb).mean())

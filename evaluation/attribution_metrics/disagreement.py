import torch

from utils.common.evaluation import spearmanr_torch


def jaccard_topk(attr_a: torch.Tensor, attr_b: torch.Tensor, k: int = 10) -> float:
    """
    Jaccard similarity of top-k features by |attr|.
    """
    a = attr_a.detach().cpu().flatten()
    b = attr_b.detach().cpu().flatten()
    D = a.numel()
    k = min(int(k), D)

    A = set(torch.topk(a.abs(), k=k).indices.tolist())
    B = set(torch.topk(b.abs(), k=k).indices.tolist())
    return float(len(A & B) / max(1, len(A | B)))


def spearman_rank_corr(attr_a: torch.Tensor, attr_b: torch.Tensor) -> float:
    """
    Spearman rank correlation between attribution vectors.
    """
    return spearmanr_torch(attr_a, attr_b)

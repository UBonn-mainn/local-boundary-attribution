from dataclasses import dataclass
from typing import Callable, Optional

import torch

from utils.common.evaluation import to_2d


@dataclass
class StabilityResult:
    lipschitz_est: float
    mean_attr_change: float


def stability_local_lipschitz(
    attr_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: float = 0.01,
    n_samples: int = 30,
    norm: str = "l2",
    seed: int = 0,
) -> StabilityResult:
    """
    Estimate local stability:
      max ||A(x)-A(x')|| / ||x-x'|| over random perturbations x' = x + noise.
    attr_fn must return attributions with same shape as x.
    """
    torch.manual_seed(seed)
    x = to_2d(x)
    base_attr = to_2d(attr_fn(x)).detach()

    ratios = []
    changes = []
    for _ in range(n_samples):
        noise = eps * torch.randn_like(x)
        x2 = x + noise
        a2 = to_2d(attr_fn(x2)).detach()

        da = a2 - base_attr
        dx = x2 - x

        if norm == "l1":
            num = da.abs().sum(dim=-1)
            den = dx.abs().sum(dim=-1) + 1e-12
        else:
            num = torch.norm(da, p=2, dim=-1)
            den = torch.norm(dx, p=2, dim=-1) + 1e-12

        ratios.append((num / den).max().item())
        changes.append(num.mean().item())

    return StabilityResult(
        lipschitz_est=float(np.max(ratios)) if ratios else 0.0,
        mean_attr_change=float(np.mean(changes)) if changes else 0.0,
    )


def sensitivity_max_captum(
    runner_attribute_fn: Callable[..., torch.Tensor],
    x: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
    target: Optional[int] = None,
    perturb_radius: float = 0.02,
    n_perturb_samples: int = 50,
) -> Optional[float]:
    """
    Uses Captum's sensitivity_max if available.
    runner_attribute_fn should be a function like:
        lambda inputs: runner.attribute(inputs, baseline=..., target=..., cfg=...)
    Returns None if captum.metrics isn't available.
    """
    try:
        from captum.metrics import sensitivity_max
    except Exception:
        return None

    x = to_2d(x)
    if baseline is not None:
        baseline = to_2d(baseline)

    def expl_func(inputs: torch.Tensor) -> torch.Tensor:
        return runner_attribute_fn(inputs)

    score = sensitivity_max(
        explanation_func=expl_func,
        inputs=x,
        perturb_radius=perturb_radius,
        n_perturb_samples=n_perturb_samples,
    )
    # sensitivity_max returns tensor per sample
    return float(score.mean().detach().cpu())



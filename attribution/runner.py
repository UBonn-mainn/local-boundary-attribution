from __future__ import annotations

from typing import Optional, Dict, Any, Callable

import torch
import torch.nn as nn
from captum.attr import (
    Saliency,
    NoiseTunnel,
    IntegratedGradients,
    KernelShap,
    Lime,
)

from attribution.config import AttributionConfig


class AttributionRunner:
    """
    Unified runner for common attribution methods.
    - model: nn.Module mapping (B, ...) -> (B, num_classes) or (B,) for binary logit
    - forward_func: optional wrapper if you need to apply softmax/logits selection
    """

    def __init__(
            self,
            model: nn.Module,
            device: Optional[torch.device] = None,
            forward_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.model = model.eval()
        self.device = device or next(model.parameters()).device
        self.forward_func = forward_func or self.model_forward_logits

    def model_forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        # Default: return raw model outputs (logits). Captum prefers logits for classification.
        return self.model(x)

    @torch.no_grad()
    def predict_label(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns argmax class index for each sample in batch.
        """
        logits = self.forward_func(x)
        if logits.ndim == 1:
            # binary scalar logit -> label 1 if logit>0 else 0
            return (logits > 0).long()
        return logits.argmax(dim=-1)

    def attribute(
            self,
            x: torch.Tensor,
            target: Optional[int] = None,
            baseline: Optional[torch.Tensor] = None,
            cfg: Optional[AttributionConfig] = None,
            extra: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Compute attributions for a single input (or batch) x.
        Returns tensor with same shape as x.
        """
        cfg = cfg or AttributionConfig()
        extra = extra or {}

        x = x.to(self.device)
        if baseline is not None:
            baseline = baseline.to(self.device)

        # If user didn't specify target, explain the model's predicted class per sample.
        if target is None:
            with torch.no_grad():
                pred = self.predict_label(x)
            # Captum target can be:
            # - int for same target across batch
            # - tensor/list for per-sample targets
            target_for_captum = pred
        else:
            target_for_captum = target

        if cfg.method == "vg":
            attr = self._vanilla_grad(x, target_for_captum)

        elif cfg.method == "sg":
            attr = self._smooth_grad(x, target_for_captum, cfg)

        elif cfg.method == "ig":
            if baseline is None:
                baseline = torch.zeros_like(x)
            attr = self._integrated_gradients(x, baseline, target_for_captum, cfg)

        elif cfg.method == "ks":
            if baseline is None:
                baseline = torch.zeros_like(x)
            attr = self._kernel_shap(x, baseline, target_for_captum, cfg)

        elif cfg.method == "lime":
            if baseline is None:
                baseline = torch.zeros_like(x)
            attr = self._lime(x, baseline, target_for_captum, cfg)

        else:
            raise ValueError(f"Unknown attribution method: {cfg.method}")

        if cfg.abs_attributions:
            attr = attr.abs()

        return attr

    # ---------------- Methods ----------------

    def _vanilla_grad(self, x: torch.Tensor, target) -> torch.Tensor:
        sal = Saliency(self.forward_func)
        return sal.attribute(x, target=target)

    def _smooth_grad(self, x: torch.Tensor, target, cfg: AttributionConfig) -> torch.Tensor:
        sal = Saliency(self.forward_func)
        nt = NoiseTunnel(sal)
        # stdevs is absolute std; scale your inputs if needed (e.g., normalize to [0,1])
        return nt.attribute(
            x,
            target=target,
            nt_type=cfg.sg_nt_type,
            n_samples=cfg.sg_n_samples,
            stdevs=cfg.sg_stdevs,
        )

    def _integrated_gradients(
            self,
            x: torch.Tensor,
            baseline: torch.Tensor,
            target,
            cfg: AttributionConfig,
    ) -> torch.Tensor:
        ig = IntegratedGradients(self.forward_func)
        return ig.attribute(
            x,
            baselines=baseline,
            target=target,
            n_steps=cfg.ig_steps,
            internal_batch_size=cfg.internal_batch_size,
        )

    def _kernel_shap(
            self,
            x: torch.Tensor,
            baseline: torch.Tensor,
            target,
            cfg: AttributionConfig,
    ) -> torch.Tensor:
        ks = KernelShap(self.forward_func)
        return ks.attribute(
            x,
            baselines=baseline,
            target=target,
            n_samples=cfg.perturb_samples,
            feature_mask=cfg.feature_mask,
            perturbations_per_eval=cfg.perturbations_per_eval,
        )

    def _lime(
            self,
            x: torch.Tensor,
            baseline: torch.Tensor,
            target,
            cfg: AttributionConfig,
    ) -> torch.Tensor:
        lime = Lime(self.forward_func)
        return lime.attribute(
            x,
            baselines=baseline,
            target=target,
            n_samples=cfg.perturb_samples,
            feature_mask=cfg.feature_mask,
            perturbations_per_eval=cfg.perturbations_per_eval,
        )

from dataclasses import dataclass
from typing import Literal, Optional

import torch

AttributionName = Literal["vg", "sg", "ig", "ks", "lime"]

@dataclass
class AttributionConfig:
    method: AttributionName = "ig"

    # Common
    abs_attributions: bool = False  # often used for comparison; keep sign if you need directionality
    internal_batch_size: Optional[int] = None

    # SmoothGrad / NoiseTunnel
    sg_nt_type: str = "smoothgrad"          # captum NoiseTunnel types: "smoothgrad", "smoothgrad_sq", "vargrad"
    sg_n_samples: int = 50
    sg_stdevs: float = 0.1                 # noise std as fraction of (max-min) if you normalize inputs; else absolute scale

    # Integrated Gradients
    ig_steps: int = 200

    # KernelSHAP / LIME
    perturb_samples: int = 300              # n_samples in captum
    feature_mask: Optional[torch.Tensor] = None  # for images/superpixels or grouped tabular features

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class GSOracleResult:
    x_start: np.ndarray
    x_enemy: np.ndarray
    x_boundary: np.ndarray
    radius_found: float
    success: bool
    meta: Dict[str, Any]


class GrowingSpheresOracle:
    def __init__(
        self,
        model: torch.nn.Module,
        n_directions: int = 200,
        r_init: float = 0.01,
        r_step: float = 0.01,
        r_max: float = 2.0,
        boundary_bisect_steps: int = 30,
        clamp: Optional[tuple[float, float]] = None,
        device: Optional[torch.device] = None,
        seed: int = 0,
    ):
        self.model = model
        self.n_directions = int(n_directions)
        self.r_init = float(r_init)
        self.r_step = float(r_step)
        self.r_max = float(r_max)
        self.boundary_bisect_steps = int(boundary_bisect_steps)
        self.clamp = clamp
        self.device = device or torch.device("cpu")
        self.rng = np.random.default_rng(seed)

        self.model.to(self.device)
        self.model.eval()

        logger.debug(
            "Initialized GrowingSpheresOracle(n_directions=%s, r_init=%s, r_step=%s, r_max=%s, bisect=%s, clamp=%s, device=%s)",
            self.n_directions, self.r_init, self.r_step, self.r_max, self.boundary_bisect_steps, self.clamp, self.device
        )

    @torch.no_grad()
    def _label(self, x_np: np.ndarray) -> int:
        x_t = torch.tensor(x_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.model(x_t)
        return int(torch.argmax(logits, dim=-1).item())

    def _bisect_to_boundary(self, x0: np.ndarray, x1: np.ndarray, y0: int) -> np.ndarray:
        lo, hi = x0.copy(), x1.copy()
        for k in range(self.boundary_bisect_steps):
            mid = 0.5 * (lo + hi)
            if self._label(mid) == y0:
                lo = mid
            else:
                hi = mid
            logger.debug("GS bisection step %d/%d", k + 1, self.boundary_bisect_steps)
        return 0.5 * (lo + hi)

    def _sample_unit_directions(self, d: int) -> np.ndarray:
        v = self.rng.normal(size=(self.n_directions, d)).astype(np.float32)
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        return v

    def find_boundary(self, x: np.ndarray, y: Optional[int] = None) -> GSOracleResult:
        x = np.asarray(x, dtype=np.float32)
        y0 = int(y) if y is not None else self._label(x)
        d = x.shape[0]

        logger.debug("GS oracle start (y0=%s, dim=%s)", y0, d)

        dirs = self._sample_unit_directions(d)

        r = self.r_init
        best_enemy = None
        best_r = None

        while r <= self.r_max:
            candidates = x[None, :] + r * dirs  # (n, d)

            if self.clamp is not None:
                lo, hi = self.clamp
                candidates = np.clip(candidates, lo, hi)

            # Find first flip (or you can search for closest flip; keeping it simple + fast)
            for i in range(candidates.shape[0]):
                if self._label(candidates[i]) != y0:
                    best_enemy = candidates[i].copy()
                    best_r = r
                    break

            logger.debug("GS radius r=%.5f checked (found=%s)", r, best_enemy is not None)

            if best_enemy is not None:
                logger.info("GS found flip at radius r=%.5f (y0=%s)", best_r, y0)
                break

            r += self.r_step

        if best_enemy is None:
            logger.warning("GS failed to find flip up to r_max=%.5f (y0=%s)", self.r_max, y0)
            return GSOracleResult(
                x_start=x,
                x_enemy=x.copy(),
                x_boundary=x.copy(),
                radius_found=float("inf"),
                success=False,
                meta={"method": "growing_spheres_oracle"},
            )

        x_boundary = self._bisect_to_boundary(x, best_enemy, y0)
        logger.info("GS boundary refined with %d bisection steps", self.boundary_bisect_steps)

        return GSOracleResult(
            x_start=x,
            x_enemy=best_enemy,
            x_boundary=x_boundary,
            radius_found=float(best_r),
            success=True,
            meta={
                "method": "growing_spheres_oracle",
                "n_directions": self.n_directions,
                "r_init": self.r_init,
                "r_step": self.r_step,
                "r_max": self.r_max,
                "bisect_steps": self.boundary_bisect_steps,
                "clamp": self.clamp,
            },
        )

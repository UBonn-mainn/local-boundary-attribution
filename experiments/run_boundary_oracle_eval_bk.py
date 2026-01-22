from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch

from boundary_search.fgsm import FGSMBoundarySearch
from evaluation.growing_spheres_oracle import GrowingSpheresOracle
from evaluation.area_compare import compute_area_fgsm_vs_gs
from evaluation.angle_metrics import angle_at_x_degrees

from utils.data.dataset_utils import load_dataset_from_csv
from utils.data.load_model import load_model
from utils.visualization.oracle_eval import (
    plot_2d_boundary_comparison,
    plot_distance_summaries,
)

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,  # switch to DEBUG when you need details
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def main():
    parser = argparse.ArgumentParser()

    # IO
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--save_path", type=str, default="results/eval/fgsm_vs_gs.csv")
    parser.add_argument("--max_points", type=int, default=200)

    # Visualization
    parser.add_argument("--vis_dir", type=str, default="results/eval/vis")
    parser.add_argument("--vis_points", type=int, default=200, help="How many per-point plots to save (2D only).")

    # Area comparison
    parser.add_argument("--area_samples", type=int, default=4000, help="Monte Carlo samples inside GS ball.")

    # FGSM params (defaults tuned for your linearly separable 2D data)
    parser.add_argument("--fgsm_step", type=float, default=0.1)
    parser.add_argument("--fgsm_max_steps", type=int, default=80)
    parser.add_argument("--fgsm_bisect_steps", type=int, default=20)

    # GS oracle params (defaults tuned for your linearly separable 2D data)
    parser.add_argument("--gs_dirs", type=int, default=128)
    parser.add_argument("--gs_r_init", type=float, default=0.1)
    parser.add_argument("--gs_r_step", type=float, default=0.01)
    parser.add_argument("--gs_r_max", type=float, default=8.0)
    parser.add_argument("--gs_bisect_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ---------------- Load data + model ----------------
    X, y = load_dataset_from_csv(args.data_path)  # X: (N,d), y: (N,)
    model = load_model(args.model_path, model_type=args.model_type)
    model = model.to(device)
    model.eval()

    # ---------------- Methods ----------------
    fgsm = FGSMBoundarySearch(
        model=model,
        step_size=args.fgsm_step,
        max_steps=args.fgsm_max_steps,
        boundary_bisect_steps=args.fgsm_bisect_steps,
        clamp=None,  # IMPORTANT: no (0,1) clamp for this dataset
        device=device,
    )

    gs_oracle = GrowingSpheresOracle(
        model=model,
        n_directions=args.gs_dirs,
        r_init=args.gs_r_init,
        r_step=args.gs_r_step,
        r_max=args.gs_r_max,
        boundary_bisect_steps=args.gs_bisect_steps,
        clamp=None,
        device=device,
        seed=args.seed,
    )

    # ---------------- Output dirs ----------------
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.vis_dir).mkdir(parents=True, exist_ok=True)

    # ---------------- Eval loop ----------------
    n = min(len(X), args.max_points)
    rows: List[Dict[str, Any]] = []

    logger.info("Evaluating %d points", n)

    for i in range(n):
        x_i = np.asarray(X[i], dtype=np.float32)
        y_i = int(y[i])

        # 1) FGSM boundary point (includes refinement via bisection)
        fgsm_res = fgsm.search(x_i, y=y_i)

        # 2) GS boundary point + radius (oracle)
        gs_res = gs_oracle.find_boundary(x_i, y=y_i)

        # 3) Areas (FGSM vs GS) inside GS ball + compare
        if fgsm_res.success and gs_res.success and np.isfinite(gs_res.radius_found) and gs_res.radius_found > 0:
            area_cmp = compute_area_fgsm_vs_gs(
                model=model,
                x=x_i,
                b_fgsm=fgsm_res.x_boundary,
                b_gs=gs_res.x_boundary,
                r_gs=float(gs_res.radius_found),
                N=args.area_samples,
                seed=args.seed + i,
            )
        else:
            area_cmp = {"area_fgsm": np.nan, "area_gs": np.nan, "area_disagreement": np.nan}

        if fgsm_res.success and gs_res.success:
            angle_deg = angle_at_x_degrees(x_i, fgsm_res.x_boundary, gs_res.x_boundary)
        else:
            angle_deg = np.nan

        row = {
            "idx": i,
            "y": y_i,
            "fgsm_success": bool(fgsm_res.success),
            "gs_success": bool(gs_res.success),
            "fgsm_steps": int(fgsm_res.num_steps),
            "gs_radius_found": float(gs_res.radius_found),

            "dist_x_to_fgsm_boundary": l2(x_i, fgsm_res.x_boundary),
            "dist_x_to_gs_boundary": l2(x_i, gs_res.x_boundary),
            "dist_fgsm_boundary_to_gs_boundary": l2(fgsm_res.x_boundary, gs_res.x_boundary),

            # area metrics (NEW)
            "area_fgsm": area_cmp["area_fgsm"],
            "area_gs": area_cmp["area_gs"],
            "area_disagreement": area_cmp["area_disagreement"],

            # angle
            "angle_x_fgsm_gs_deg": angle_deg,
        }
        rows.append(row)

        # 4) Visualization (2D only)
        if X.shape[1] == 2 and i < args.vis_points:
            out_path = Path(args.vis_dir) / f"pt_{i:04d}.png"
            plot_2d_boundary_comparison(
                model=model,
                x=x_i,
                b_fgsm=fgsm_res.x_boundary if fgsm_res.success else None,
                b_gs=gs_res.x_boundary if gs_res.success else None,
                gs_radius=gs_res.radius_found if gs_res.success else None,
                save_path=out_path,
                X_train=X,
                y_train=y,
                device=device,
                title=(
                    f"idx={i} | fgsm={fgsm_res.success} | gs={gs_res.success} | "
                    f"A_f={row['area_fgsm']:.3f} | A_g={row['area_gs']:.3f} | A_Δ={row['area_disagreement']:.3f}"
                    if np.isfinite(row["area_disagreement"])
                    else f"idx={i} | fgsm={fgsm_res.success} | gs={gs_res.success}"
                ),
            )

        if (i + 1) % 25 == 0:
            logger.info("Processed %d/%d points", i + 1, n)

    # ---------------- Save report ----------------
    df = pd.DataFrame(rows)
    df.to_csv(args.save_path, index=False)

    logger.info("Saved report CSV: %s", args.save_path)
    logger.info("Summary:\n%s", df.describe(include="all"))

    # Summary histograms
    plot_distance_summaries(df, save_dir=args.vis_dir, prefix="fgsm_vs_gs")
    logger.info("Visualizations saved to: %s", args.vis_dir)


if __name__ == "__main__":
    main()

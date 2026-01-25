from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch

from boundary_search.fgsm import FGSMBoundarySearch
from evaluation.angle_metrics import angle_at_x_degrees
from evaluation.area_compare import compute_area_fgsm_vs_gs
# NEW: curve + circle area (global FGSM boundary curve + per-x circle)
from evaluation.curve_circle_area import cal_curve_circle_area
from evaluation.growing_spheres_oracle import GrowingSpheresOracle
from evaluation.ring_counts import count_classes_in_ring
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

    # Area comparison (FGSM vs GS induced hyperplanes inside GS ball)
    parser.add_argument("--area_samples", type=int, default=4000, help="Monte Carlo samples inside GS ball.")

    # Curve-vs-circle area (global FGSM curve + per-x circle)
    parser.add_argument("--curve_arc_resolution", type=int, default=200,
                        help="Number of points to discretize the circle arc.")
    parser.add_argument("--curve_use_shorter_arc", action="store_true",
                        help="Use the shorter of the two circle arcs (recommended).")
    parser.add_argument("--curve_circle_samples", type=int, default=20000, help="MC samples for curve circle area.")
    # FGSM params
    parser.add_argument("--fgsm_step", type=float, default=0.1)
    parser.add_argument("--fgsm_max_steps", type=int, default=80)
    parser.add_argument("--fgsm_bisect_steps", type=int, default=20)

    # GS oracle params
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
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

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

    # ---------------- Pass 1: run FGSM/GS and collect results ----------------
    n = min(len(X), args.max_points)
    logger.info("Evaluating %d points", n)

    # Store per-point info for a second pass (curve-based area requires global curve)
    per_point: List[Dict[str, Any]] = []
    all_fgsm_boundary_points: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []

    for i in range(n):
        x_i = np.asarray(X[i], dtype=np.float32)
        y_i = int(y[i])

        fgsm_res = fgsm.search(x_i, y=y_i)  # includes refinement (bisection)
        gs_res = gs_oracle.find_boundary(x_i, y=y_i)

        # Collect FGSM boundary points for building a global curve (only if success)
        if fgsm_res.success:
            all_fgsm_boundary_points.append(np.asarray(fgsm_res.x_boundary, dtype=np.float32))

        # Angle metric
        if fgsm_res.success and gs_res.success:
            angle_deg = angle_at_x_degrees(x_i, fgsm_res.x_boundary, gs_res.x_boundary)
        else:
            angle_deg = np.nan

        r_fgsm = float(
            np.linalg.norm(np.asarray(fgsm_res.x_boundary, dtype=np.float32) - x_i)) if fgsm_res.success else np.nan
        if fgsm_res.success and X.shape[1] == 2:
            fgsm_cc_area = cal_curve_circle_area(model, x_i, r_fgsm, device=device, n_samples=args.curve_circle_samples,
                                                 seed=args.seed + i)
        else:
            fgsm_cc_area = {"curve_circle_area": np.nan, "curve_circle_frac": np.nan}

        # Area metrics: FGSM vs GS induced hyperplanes inside GS ball
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

        if fgsm_res.success and gs_res.success and np.isfinite(gs_res.radius_found) and gs_res.radius_found > 0:
            ring = count_classes_in_ring(
                x=x_i,
                b_fgsm=fgsm_res.x_boundary,
                r_gs=float(gs_res.radius_found),
                X_ref=X,  # use your dataset points as the reference set
                y_ref=y,  # counts by dataset labels
                model=model,  # counts by model predictions too (optional)
                device=device,
            )
        else:
            ring = {"ring_n": 0, "ring_r_min": np.nan, "ring_r_max": np.nan}

        row = {
            "idx": i,
            "y": y_i,

            "fgsm_success": bool(fgsm_res.success),
            "gs_success": bool(gs_res.success),
            "fgsm_steps": int(fgsm_res.num_steps),
            "gs_radius_found": float(gs_res.radius_found),
            # "b_fgsm": np.asarray(fgsm_res.x_boundary, dtype=np.float32),
            # "b_gs": np.asarray(gs_res.x_boundary, dtype=np.float32),

            "angle_x_fgsm_gs_deg": float(angle_deg),

            "curve_circle_area": fgsm_cc_area["curve_circle_area"],
            "curve_circle_frac": fgsm_cc_area["curve_circle_frac"],

            "dist_x_to_fgsm_boundary": l2(x_i, fgsm_res.x_boundary),
            "dist_x_to_gs_boundary": l2(x_i, gs_res.x_boundary),
            "dist_fgsm_boundary_to_gs_boundary": l2(fgsm_res.x_boundary, gs_res.x_boundary),

            "area_fgsm": float(area_cmp["area_fgsm"]),
            "area_gs": float(area_cmp["area_gs"]),
            "area_disagreement": float(area_cmp["area_disagreement"]),

            "ring_n": ring.get("ring_n", 0),
            "ring_r_min": ring.get("ring_r_min", np.nan),
            "ring_r_max": ring.get("ring_r_max", np.nan),

            "ring_count_y0": ring.get("ring_count_y0", 0),
            "ring_count_y1": ring.get("ring_count_y1", 0),

            "ring_count_pred0": ring.get("ring_count_pred0", 0),
            "ring_count_pred1": ring.get("ring_count_pred1", 0),
        }
        rows.append(row)

        if (i + 1) % 25 == 0:
            logger.info("Processed %d/%d points", i + 1, n)

            # Visualization (2D only)
        i = row["idx"]
        if X.shape[1] == 2 and i < args.vis_points:
            out_path = Path(args.vis_dir) / f"pt_{i:04d}.png"
            title = (
                f"idx={i} | fgsm={row['fgsm_success']} | gs={row['gs_success']} | "
                f"angle={row['angle_x_fgsm_gs_deg']:.1f}° | "
                f"AΔ={row['area_disagreement']:.3f} | "
                f"curve∩circle area={row['curve_circle_area']:.3f}"
                if np.isfinite(row["angle_x_fgsm_gs_deg"]) and np.isfinite(row["area_disagreement"])
                else f"idx={i} | fgsm={row['fgsm_success']} | gs={row['gs_success']}"
            )

            plot_2d_boundary_comparison(
                model=model,
                x=x_i,
                b_fgsm=np.asarray(fgsm_res.x_boundary, dtype=np.float32) if row["fgsm_success"] else None,
                b_gs=np.asarray(gs_res.x_boundary, dtype=np.float32) if row["gs_success"] else None,
                gs_radius=row["gs_radius_found"] if row["gs_success"] else None,
                save_path=out_path,
                X_train=X,
                y_train=y,
                device=device,
                title=title,
                # This will draw a circle centered at x with radius ||x-b_fgsm||:
                fgsm_circle=True,
                show_soft_surface=True,
                soft_mode="p_class1",  # "p_true" | "p_class1" | "p_max"
                soft_levels=20,
                soft_alpha=0.85,
                show_hard_contour=True
            )

    # ---------------- Save report ----------------
    df = pd.DataFrame(rows)
    df.to_csv(args.save_path, index=False)

    logger.info("Saved report CSV: %s", args.save_path)
    logger.info("Summary:\n%s", df.describe(include="all"))

    # Summary histograms (distances + any numeric cols you already plot)
    plot_distance_summaries(df, save_dir=args.vis_dir, prefix="fgsm_vs_gs")
    logger.info("Visualizations saved to: %s", args.vis_dir)


if __name__ == "__main__":
    main()

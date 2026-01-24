import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from boundary_search.fgsm import fgsm_boundary_search
from boundary_search.ibs import ibs_boundary_search
from boundary_search.hybrid import HybridFGSMIBS
from boundary_search.boundary_walker import BoundaryCrawler
from boundary_search.growing_spheres import (
    growing_spheres_search,
    optimize_proximity_with_growing_spheres,
)
from evaluation import GrowingSpheresOracle
from evaluation.angle_metrics import angle_at_x_degrees
from evaluation.curve_circle_area import cal_curve_circle_area
from evaluation.decision_boundary_metrics import (
    neutrality_metric,
    proximity_metric,
    stability_metric,
    flip_correctness_metric,
)
from evaluation.decision_boundary_sphere_volume import decision_boundary_sphere_volume
from evaluation.ring_counts import count_classes_in_ring
from evaluation.visualize_decision_boundary import plot_decision_boundary
from utils.data.dataset_utils import load_dataset_from_csv
from utils.data.load_model import load_model
from utils.visualization import plot_2d_boundary_comparison

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,  # switch to DEBUG when you need details
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def main():
    parser = argparse.ArgumentParser(description="Run boundary search on a dataset.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "linear"], help="Type of model")
    parser.add_argument("--save_dir", type=str, default="results/boundary_search", help="Directory to save results")
    parser.add_argument(
        "--method",
        type=str,
        default="fgsm",
        choices=["growing_spheres", "fgsm", "ibs", "fgsm_ibs", "crawler"],
        help="Boundary search method to use (default: FGSM)",
    )
    parser.add_argument("--ibs_candidates", type=int, default=10, help="Number of candidates for IBS")
    parser.add_argument("--ibs_max_iter", type=int, default=50, help="Max iterations for IBS binary search")
    parser.add_argument("--crawler_samples", type=int, default=10, help="Samples per crawler ring")
    parser.add_argument("--crawler_iter", type=int, default=10, help="Max crawler iterations")
    parser.add_argument("--crawler_step", type=float, default=0.05, help="Crawler ring radius")
    parser.add_argument(
        "--crawler_mode",
        type=str,
        default="random",
        choices=["random", "gradient"],
        help="Crawler mode: 'random' ring or 'gradient' projection"
    )
    parser.add_argument("--step_size", type=float, default=0.01, help="FGSM step size")
    parser.add_argument("--max_iters", type=int, default=50, help="FGSM max iterations")
    parser.add_argument("--gs_initial_radius", type=float, default=0.1, help="Growing Spheres initial radius")
    parser.add_argument("--gs_step_radius", type=float, default=0.05, help="Growing Spheres radius increment")
    parser.add_argument("--gs_max_radius", type=float, default=5.0, help="Growing Spheres maximum search radius")
    parser.add_argument("--gs_samples", type=int, default=512, help="Samples per Growing Spheres shell")
    parser.add_argument(
        "--optimize_proximity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Growing Spheres to optimize FGSM baselines for proximity",
    )
    parser.add_argument("--stability_runs", type=int, default=5,
                        help="Number of runs per sample for stability evaluation")
    parser.add_argument("--stability_samples", type=int, default=20, help="Number of samples to evaluate stability on")

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

    # GS oracle params
    parser.add_argument("--gs_dirs", type=int, default=128)
    parser.add_argument("--gs_r_init", type=float, default=0.1)
    parser.add_argument("--gs_r_step", type=float, default=0.01)
    parser.add_argument("--gs_r_max", type=float, default=8.0)
    parser.add_argument("--gs_bisect_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # Setup paths
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load Data
    print(f"Loading data from {args.data_path}...")
    X, y = load_dataset_from_csv(args.data_path)

    # Load Model
    print(f"Loading model from {args.model_path} ({args.model_type})...")
    model = load_model(args.model_path, input_dim=X.shape[1], model_type=args.model_type)

    device = next(model.parameters()).device

    def run_search(x_input: torch.Tensor):
        method_used = args.method
        if args.method == "growing_spheres":
            b_point, success = growing_spheres_search(
                model,
                x_input,
                initial_radius=args.gs_initial_radius,
                step_radius=args.gs_step_radius,
                max_radius=args.gs_max_radius,
                clamp=None,
                refine_steps=12,
            )
        elif args.method == "ibs":
            # IBS expects numpy input
            x_np = x_input.detach().cpu().numpy()
            b_point_np, success = ibs_boundary_search(
                model=model,
                x=x_np,
                X_train=X,
                y_train=y,
                num_candidates=args.ibs_candidates
            )
            b_point = torch.from_numpy(b_point_np).to(device)

        elif args.method == "fgsm_ibs":
            # Hybrid FGSM+IBS
            x_np = x_input.detach().cpu().numpy()

            fgsm_params = {
                "step_size": args.step_size,
                "max_steps": args.max_iters,
                "clamp": None,  # Synthetic data
            }
            ibs_params = {
                "num_candidates": args.ibs_candidates,
                "max_iterations": args.ibs_max_iter,
            }

            searcher = HybridFGSMIBS(
                model=model,
                X_train=X,
                y_train=y,
                device=device,
                fgsm_params=fgsm_params,
                ibs_params=ibs_params
            )

            res = searcher.search(x_np)
            b_point = torch.from_numpy(res.x_boundary).to(device)
            success = res.success

        elif args.method == "crawler":
            # Boundary Crawler
            x_np = x_input.detach().cpu().numpy()

            # FGSM params for initialization
            fgsm_params = {
                "step_size": args.step_size,
                "max_steps": args.max_iters,
                "clamp": None
            }
            # Crawler params
            crawl_params = {
                "mode": args.crawler_mode,
                "max_iterations": args.crawler_iter,
                "num_samples": args.crawler_samples,
                "step_size": args.crawler_step,
                "bisection_steps": 10
            }

            crawler = BoundaryCrawler(
                model=model,
                device=device,
                fgsm_params=fgsm_params,
                crawl_params=crawl_params
            )

            res = crawler.search(x_np)
            b_point = torch.from_numpy(res.x_boundary).to(device)
            success = res.success

        else:
            # Note: clamp is None for this synthetic data, but typically (0,1) for images
            b_point, success = fgsm_boundary_search(
                model,
                x_input,
                step_size=args.step_size,
                max_iters=args.max_iters,
                clamp=None,
                refine_steps=10,
            )

            if success and args.optimize_proximity:
                optimized_point, optimized = optimize_proximity_with_growing_spheres(
                    model,
                    x_input,
                    b_point,
                    initial_radius=args.gs_initial_radius,
                    step_radius=args.gs_step_radius,
                    n_samples=args.gs_samples,
                    clamp=None,
                    refine_steps=8,
                )

                if optimized:
                    b_point = optimized_point
                    method_used = "fgsm+growing_spheres"

        return b_point, success, method_used

    # Run Search
    print("Running boundary search...")
    boundary_points = []
    rows: List[Dict[str, Any]] = []

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

    for i in tqdm(range(len(X))):

        x_input = torch.tensor(X[i], dtype=torch.float32, device=device)
        y_i = y[i]

        crawler_b_point, crawler_success, method_used = run_search(x_input)

        b_point_np = crawler_b_point.detach().cpu().numpy()
        fgsm_dist = np.linalg.norm(b_point_np - X[i])

        if crawler_success:
            boundary_points.append(b_point_np)

        gs_res = gs_oracle.find_boundary(x_input, y=y_i)

        # Angle metric
        if crawler_success and gs_res.success:
            angle_deg = angle_at_x_degrees(x_input, crawler_b_point, gs_res.x_boundary)
        else:
            angle_deg = np.nan
        # volume/area
        if crawler_success:
            decision_boundary_sphere_area = decision_boundary_sphere_volume(model, x_input, fgsm_dist, device=device,
                                                                            n_samples=args.curve_circle_samples,
                                                                            seed=args.seed + i)
        else:
            decision_boundary_sphere_area = {"decision_boundary_sphere_area": np.nan,
                                             "decision_boundary_sphere_frac": np.nan}

        # annulus point count
        if crawler_success and gs_res.success and np.isfinite(gs_res.radius_found) and gs_res.radius_found > 0:
            ring = count_classes_in_ring(
                x=x_input,
                b_fgsm=crawler_b_point,
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

            "crawler_success": bool(crawler_success),
            "gs_success": bool(gs_res.success),

            "gs_radius_found": float(gs_res.radius_found),
            "angle_x_fgsm_gs_deg": float(angle_deg),

            "curve_circle_area": decision_boundary_sphere_area["decision_boundary_sphere_area"],
            "curve_circle_frac": decision_boundary_sphere_area["decision_boundary_sphere_frac"],

            "dist_x_to_fgsm_boundary": l2(x_input, crawler_b_point),
            "dist_x_to_gs_boundary": l2(x_input, gs_res.x_boundary),
            "dist_fgsm_boundary_to_gs_boundary": l2(crawler_b_point, gs_res.x_boundary),

            "ring_n": ring.get("ring_n", 0), # all class
            "ring_r_min": ring.get("ring_r_min", np.nan),
            "ring_r_max": ring.get("ring_r_max", np.nan),

            "ring_count_y0": ring.get("ring_count_y0", 0), # each class
            "ring_count_y1": ring.get("ring_count_y1", 0), # each class

            "ring_count_pred0": ring.get("ring_count_pred0", 0),
            "ring_count_pred1": ring.get("ring_count_pred1", 0),
        }
        rows.append(row)


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
                x=x_input,
                b_fgsm=np.asarray(crawler_b_point, dtype=np.float32) if row["fgsm_success"] else None,
                b_gs=np.asarray(gs_res.x_boundary, dtype=np.float32) if row["gs_success"] else None,
                gs_radius=row["gs_radius_found"] if row["gs_success"] else None,
                save_path=out_path,
                X_train=X,
                y_train=y,
                device=device,
                title=title,
                # This will draw a circle centered at x with radius ||x-b_fgsm||:
                fgsm_circle=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(args.save_path, index=False)

    logger.info("Saved report CSV: %s", args.save_path)
    logger.info("Summary:\n%s", df.describe(include="all"))


if __name__ == "__main__":
    main()

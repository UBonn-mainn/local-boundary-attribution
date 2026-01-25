import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from boundary_search.boundary_walker import BoundaryCrawler
from evaluation import GrowingSpheresOracle
from evaluation.angle_metrics import angle_at_x_degrees
from evaluation.decision_boundary_sphere_volume import decision_boundary_sphere_volume
from evaluation.ring_counts import count_classes_in_ring
from utils.data.dataset_utils import load_dataset_from_csv
from utils.data.load_model import load_model
from utils.visualization import plot_2d_boundary_comparison

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def main():
    parser = argparse.ArgumentParser(description="Run crawler boundary search evaluation.")
    
    # Required paths
    parser.add_argument("--data_path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "linear"], help="Type of model")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of output classes")
    parser.add_argument("--save_dir", type=str, default="results/boundary_search", help="Directory to save results")

    # Crawler parameters
    parser.add_argument("--crawler_samples", type=int, default=10, help="Samples per crawler ring")
    parser.add_argument("--crawler_iter", type=int, default=50, help="Max crawler iterations")
    parser.add_argument("--crawler_step", type=float, default=0.1, help="Crawler ring radius")
    parser.add_argument("--crawler_mode", type=str, default="random", choices=["random", "gradient"],
                        help="Crawler mode: 'random' ring or 'gradient' projection")
    
    # FGSM init parameters (used by crawler for initial boundary point)
    parser.add_argument("--fgsm_step_size", type=float, default=0.005, help="FGSM step size for crawler init")
    parser.add_argument("--fgsm_max_iters", type=int, default=100, help="FGSM max iterations for crawler init")

    # Visualization
    parser.add_argument("--vis_dir", type=str, default="results/eval/vis")
    parser.add_argument("--vis_points", type=int, default=200, help="How many per-point plots to save (2D only).")

    # GS oracle params (for ground truth comparison)
    parser.add_argument("--gs_dirs", type=int, default=128, help="Number of directions for GS oracle")
    parser.add_argument("--gs_r_init", type=float, default=0.1, help="GS oracle initial radius")
    parser.add_argument("--gs_r_step", type=float, default=0.01, help="GS oracle radius step")
    parser.add_argument("--gs_r_max", type=float, default=8.0, help="GS oracle max radius")
    parser.add_argument("--gs_bisect_steps", type=int, default=25, help="GS oracle bisection steps")
    
    # MC sampling for sphere volume
    parser.add_argument("--sphere_samples", type=int, default=20000, help="MC samples for sphere volume calculation")
    
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
    model = load_model(args.model_path, input_dim=X.shape[1], model_type=args.model_type, num_classes=args.num_classes)

    device = next(model.parameters()).device

    # Setup crawler
    fgsm_params = {
        "step_size": args.fgsm_step_size,
        "max_steps": args.fgsm_max_iters,
        "clamp": None
    }
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

    def run_search(x_input: torch.Tensor):
        x_np = x_input.detach().cpu().numpy()
        res = crawler.search(x_np)
        b_point = torch.from_numpy(res.x_boundary).to(device)
        return b_point, res.success

    # Run Search
    print("Running crawler boundary search...")
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
        x_np = X[i]  # Already numpy
        x_input = torch.tensor(x_np, dtype=torch.float32, device=device)
        y_i = y[i]

        crawler_b_point, crawler_success = run_search(x_input)

        b_point_np = crawler_b_point.detach().cpu().numpy()
        crawler_dist = np.linalg.norm(b_point_np - x_np)

        if crawler_success:
            boundary_points.append(b_point_np)

        gs_res = gs_oracle.find_boundary(x_np, y=y_i)

        # Angle metric (use numpy arrays)
        if crawler_success and gs_res.success:
            angle_deg = angle_at_x_degrees(x_np, b_point_np, gs_res.x_boundary)
        else:
            angle_deg = np.nan

        # Decision boundary sphere volume/area
        if crawler_success:
            sphere_result = decision_boundary_sphere_volume(
                model, x_np, crawler_dist, device=device,
                n_samples=args.sphere_samples, seed=args.seed + i
            )
        else:
            sphere_result = {"red_frac": np.nan, "red_vol": np.nan}

        # Annulus point count
        if crawler_success and gs_res.success and np.isfinite(gs_res.radius_found) and gs_res.radius_found > 0:
            ring = count_classes_in_ring(
                x=x_np,
                b_fgsm=b_point_np,
                r_gs=float(gs_res.radius_found),
                X_ref=X,
                y_ref=y,
                model=model,
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
            "angle_x_crawler_gs_deg": float(angle_deg),

            "sphere_frac": sphere_result["red_frac"],
            "sphere_vol": sphere_result["red_vol"],

            "dist_x_to_crawler": l2(x_np, b_point_np),
            "dist_x_to_gs": l2(x_np, gs_res.x_boundary),
            "dist_crawler_to_gs": l2(b_point_np, gs_res.x_boundary),

            "ring_n": ring.get("ring_n", 0),
            "ring_r_min": ring.get("ring_r_min", np.nan),
            "ring_r_max": ring.get("ring_r_max", np.nan),

            "ring_count_y0": ring.get("ring_count_y0", 0),
            "ring_count_y1": ring.get("ring_count_y1", 0),

            "ring_count_pred0": ring.get("ring_count_pred0", 0),
            "ring_count_pred1": ring.get("ring_count_pred1", 0),
        }
        rows.append(row)

        # Visualization for 2D data
        if X.shape[1] == 2 and i < args.vis_points:
            vis_dir = Path(args.vis_dir)
            vis_dir.mkdir(parents=True, exist_ok=True)
            out_path = vis_dir / f"pt_{i:04d}.png"
            title = (
                f"idx={i} | crawler={row['crawler_success']} | gs={row['gs_success']} | "
                f"angle={row['angle_x_crawler_gs_deg']:.1f}° | "
                f"sphere_frac={row['sphere_frac']:.3f}"
                if np.isfinite(row["angle_x_crawler_gs_deg"]) and np.isfinite(row["sphere_frac"])
                else f"idx={i} | crawler={row['crawler_success']} | gs={row['gs_success']}"
            )

            plot_2d_boundary_comparison(
                model=model,
                x=x_np,
                b_fgsm=b_point_np if row["crawler_success"] else None,
                b_gs=np.asarray(gs_res.x_boundary, dtype=np.float32) if row["gs_success"] else None,
                gs_radius=row["gs_radius_found"] if row["gs_success"] else None,
                save_path=out_path,
                X_train=X,
                y_train=y,
                device=device,
                title=title,
                fgsm_circle=True,
            )

    df = pd.DataFrame(rows)
    report_path = save_dir / "report.csv"
    df.to_csv(report_path, index=False)

    logger.info("Saved report CSV: %s", report_path)
    logger.info("Summary:\n%s", df.describe(include="all"))


if __name__ == "__main__":
    main()

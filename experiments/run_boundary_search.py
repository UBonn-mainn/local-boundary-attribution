import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from boundary_search.fgsm import fgsm_boundary_search
from boundary_search.growing_spheres import (
    growing_spheres_search,
    optimize_proximity_with_growing_spheres,
)
from evaluation.decision_boundary_metrics import (
    neutrality_metric,
    proximity_metric,
    stability_metric,
    flip_correctness_metric,
)
from evaluation.visualize_decision_boundary import plot_decision_boundary
from utils.data.dataset_utils import load_dataset_from_csv
from utils.data.load_model import load_model


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
        choices=["growing_spheres", "fgsm"],
        help="Boundary search method to use (default: FGSM)",
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
                n_samples=args.gs_samples,
                clamp=None,
                refine_steps=12,
            )
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
    results = []
    boundary_points = []
    neutrality_vals = []
    proximity_vals = []
    flip_correct_vals = []
    success_flags = []

    for i in tqdm(range(len(X))):
        # # x_input = torch.tensor(X[i], dtype=torch.float32)
        # #
        # # # Run FGSM
        # # # Note: clamp is None for this synthetic data, but typically (0,1) for images
        # # b_point, success = fgsm_boundary_search(
        # #     model,
        # #     x_input,
        # #     step_size=args.step_size,
        # #     max_iters=args.max_iters,
        # #     clamp=None,
        # #     refine_steps=10
        # # )
        #
        # b_point_np = b_point.numpy()
        x_input = torch.tensor(X[i], dtype=torch.float32, device=device)

        b_point, success, method_used = run_search(x_input)

        b_point_np = b_point.detach().cpu().numpy()
        dist = np.linalg.norm(b_point_np - X[i])

        if success:
            neutrality = neutrality_metric(
                model=model,
                x_boundary=b_point,
                x_original=x_input,
                cls_mode="original",
            )
            proximity = proximity_metric(
                x_original=x_input,
                x_boundary=b_point,
                p=2.0,
            )
            flip_info = flip_correctness_metric(
                model=model,
                x_original=x_input,
                x_boundary=b_point,
                eps=1e-2,
            )
            flip_correct = bool(flip_info["flip_correct"])
        else:
            neutrality = float("nan")
            proximity = float("nan")
            flip_correct = False

        results.append({
            "original_index": i,
            "original_class": y[i],
            "success": success,
            "boundary_x1": b_point_np[0],
            "boundary_x2": b_point_np[1],
            "distance": dist,
            "method": method_used,
            "neutrality": neutrality,
            "proximity_l2": proximity,
            "flip_correct": flip_correct,
        })

        neutrality_vals.append(neutrality)
        proximity_vals.append(proximity)
        flip_correct_vals.append(flip_correct)
        success_flags.append(success)

        if success:
            boundary_points.append(b_point_np)
    # print("test", boundary_points)
    # Save Results
    df_results = pd.DataFrame(results)
    csv_path = save_dir / "boundary_points.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    print(f"Success Rate: {df_results['success'].mean() * 100:.2f}%")
    print(f"Mean Distance: {df_results[df_results['success']]['distance'].mean():.4f}")

    # Evaluation metrics
    success_arr = np.array(success_flags, dtype=bool)
    neutrality_arr = np.array(neutrality_vals, dtype=float)
    proximity_arr = np.array(proximity_vals, dtype=float)
    flip_arr = np.array(flip_correct_vals, dtype=bool)

    mask = success_arr & ~np.isnan(neutrality_arr) & ~np.isnan(proximity_arr)

    mean_neutrality = float(np.nanmean(neutrality_arr[mask])) if mask.any() else float("nan")
    median_neutrality = float(np.nanmedian(neutrality_arr[mask])) if mask.any() else float("nan")
    mean_proximity = float(np.nanmean(proximity_arr[mask])) if mask.any() else float("nan")
    median_proximity = float(np.nanmedian(proximity_arr[mask])) if mask.any() else float("nan")
    flip_rate = float(flip_arr[mask].mean()) if mask.any() else float("nan")

    stability_indices = np.random.choice(len(X), size=min(args.stability_samples, len(X)), replace=False)
    stability_results = []
    for idx in stability_indices:
        x_stab = torch.tensor(X[idx], dtype=torch.float32, device=device)
        stab = stability_metric(
            model=model,
            x=x_stab,
            search_fn=lambda m, x: run_search(x)[:2],
            n_runs=args.stability_runs,
            device=device,
        )
        stability_results.append(stab)

    if stability_results:
        mean_stability_success = float(np.mean([s.success_rate for s in stability_results]))

        pairwise_vals = [s.mean_pairwise_distance for s in stability_results if not np.isnan(s.mean_pairwise_distance)]
        mean_pairwise = float(np.mean(pairwise_vals)) if pairwise_vals else float("nan")

        std_to_mean_vals = [s.std_to_mean for s in stability_results if not np.isnan(s.std_to_mean)]
        mean_std_to_mean = float(np.mean(std_to_mean_vals)) if std_to_mean_vals else float("nan")
    else:
        mean_stability_success = float("nan")
        mean_pairwise = float("nan")
        mean_std_to_mean = float("nan")

    summary_path = save_dir / "boundary_metrics_summary.csv"
    pd.DataFrame(
        [
            ("success_rate", success_arr.mean()),
            ("mean_neutrality", mean_neutrality),
            ("median_neutrality", median_neutrality),
            ("mean_proximity_l2", mean_proximity),
            ("median_proximity_l2", median_proximity),
            ("flip_correct_rate", flip_rate),
            ("stability_success_rate", mean_stability_success),
            ("stability_mean_pairwise_dist", mean_pairwise),
            ("stability_std_to_mean", mean_std_to_mean),
        ],
        columns=["metric", "value"],
    ).to_csv(summary_path, index=False)
    print(f"Saved summary metrics to {summary_path}")

    visualize_results(
        model=model,
        X=X,
        y=y,
        save_dir=save_dir,
        boundary_points=boundary_points,
        success_mask=mask,
        neutrality_arr=neutrality_arr,
        proximity_arr=proximity_arr,
        metrics_summary={
            "success_rate": success_arr.mean(),
            "mean_neutrality": mean_neutrality,
            "median_neutrality": median_neutrality,
            "mean_proximity_l2": mean_proximity,
            "median_proximity_l2": median_proximity,
            "flip_correct_rate": flip_rate,
            "stability_success_rate": mean_stability_success,
            "stability_mean_pairwise_dist": mean_pairwise,
            "stability_std_to_mean": mean_std_to_mean,
        },
        model_type=args.model_type,
    )


def visualize_results(
        model,
        X: np.ndarray,
        y: np.ndarray,
        save_dir: Path,
        boundary_points,
        success_mask: np.ndarray,
        neutrality_arr: np.ndarray,
        proximity_arr: np.ndarray,
        metrics_summary: dict,
        model_type: str,
):
    """Generate plots and a compact report for boundary search runs."""

    outputs = []
    if boundary_points:
        bp_array = np.array(boundary_points)
        plot_path = save_dir / f"boundary_vis_{model_type}.png"
        plot_decision_boundary(
            model,
            X,
            y,
            output_path = str(plot_path),
            extra_scatter = (bp_array, "Boundary Points"),
        )
        outputs.append(plot_path)

    if success_mask.any():
        neutrality_path = save_dir / "neutrality_histogram.png"
        plt.figure(figsize=(6, 4))
        plt.hist(neutrality_arr[success_mask], bins=30)
        plt.xlabel("|P_c(x') - 0.5|")
        plt.ylabel("Count")
        plt.title("Neutrality Distribution")
        plt.tight_layout()
        plt.savefig(neutrality_path, dpi=150)
        plt.close()
        outputs.append(neutrality_path)

        proximity_path = save_dir / "proximity_histogram.png"
        plt.figure(figsize=(6, 4))
        plt.hist(proximity_arr[success_mask], bins=30)
        plt.xlabel("||x - x'||_2")
        plt.ylabel("Count")
        plt.title("L2 Proximity Distribution")
        plt.tight_layout()
        plt.savefig(proximity_path, dpi=150)
        plt.close()
        outputs.append(proximity_path)

        if X.shape[1] == 2 and boundary_points:
            scatter_path = save_dir / "samples_vs_boundaries.png"
            X_succ = X[success_mask]

            # boundary_points stores successful results in the same order we iterate the data,
            # which matches the order of X[success_mask]. Ensure lengths align before plotting.
            bnd_succ = np.array(boundary_points)
            if len(bnd_succ) != len(X_succ):
                print("Skipping scatter plot because boundary points do not align with successes.")
            else:
                plt.figure(figsize=(7, 6))
                plt.scatter(X[y == 0, 0], X[y == 0, 1], c="blue", alpha=0.3, label="Class 0", edgecolors="none")
                plt.scatter(X[y == 1, 0], X[y == 1, 1], c="red", alpha=0.3, label="Class 1", edgecolors="none")
                for s, b in zip(X_succ, bnd_succ):
                    plt.plot([s[0], b[0]], [s[1], b[1]], "k--", linewidth=1, alpha=0.7)
                plt.scatter(X_succ[:, 0], X_succ[:, 1], c="green", edgecolors="k", label="Samples")
                plt.scatter(bnd_succ[:, 0], bnd_succ[:, 1], c="yellow", edgecolors="k", marker="X", s=80,
                            label="Boundary points")
                plt.xlabel("Feature 1")
                plt.ylabel("Feature 2")
                plt.title("Samples and Decision-Boundary Points")
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(scatter_path, dpi=150)
                plt.close()
                outputs.append(scatter_path)

    report_path = save_dir / "boundary_metrics_overview.png"
    _plot_metrics_overview(metrics_summary, report_path)
    outputs.append(report_path)

    if outputs:
        print("Saved visual reports:")
        for path in outputs:
            print(f" - {path}")
    else:
        print("No visuals generated (no successful boundary points to plot).")


def _plot_metrics_overview(metrics_summary: dict, output_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axis("off")

    text_lines = ["Boundary Search Metrics Overview", ""]
    for key, value in metrics_summary.items():
        display_val = "NaN" if np.isnan(value) else f"{value:.4f}"
        text_lines.append(f"{key.replace('_', ' ').title()}: {display_val}")

    ax.text(0.02, 0.98, "\n".join(text_lines), va="top", ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()

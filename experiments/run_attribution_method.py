from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from attribution.config import AttributionConfig
from attribution.runner import AttributionRunner
from boundary_search.boundary_walker import BoundaryCrawler
from evaluation.angle_metrics import angle_at_x_degrees
from evaluation.attribution_metrics.baseline_effect import attribution_delta_norm
from evaluation.attribution_metrics.boundary_alignment import boundary_alignment_cosine
from evaluation.attribution_metrics.disagreement import jaccard_topk, spearman_rank_corr
from evaluation.attribution_metrics.fidelity import fidelity_deletion_insertion
from evaluation.attribution_metrics.ground_truth import true_gradient_attribution, cosine_with_true_gradient
from evaluation.attribution_metrics.stability import stability_local_lipschitz
from evaluation.decision_boundary_sphere_volume import decision_boundary_sphere_volume
from evaluation.growing_spheres_oracle import GSOracleResult
from evaluation.ring_counts import count_classes_in_ring

from utils.data.dataset_utils import load_dataset
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


def _np_to_torch_float(x, device) -> torch.Tensor:
    """
    Robust conversion for numpy arrays that might be dtype=object.
    """
    arr = np.asarray(x)
    if arr.dtype == object:
        arr = np.asarray(arr.tolist(), dtype=np.float32)
    else:
        arr = arr.astype(np.float32, copy=False)
    return torch.tensor(arr, dtype=torch.float32, device=device)


def compute_attr(
        runner: AttributionRunner,
        x: torch.Tensor,  # (d,) or (1,d)
        baseline: torch.Tensor,  # (d,) or (1,d)
        target: int,
        cfg: AttributionConfig,
) -> torch.Tensor:
    """
    Returns attributions shaped (1, d).
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if baseline.ndim == 1:
        baseline = baseline.unsqueeze(0)

    attr = runner.attribute(x=x, baseline=baseline, target=target, cfg=cfg)
    return attr.detach()


def main():
    parser = argparse.ArgumentParser(description="Run crawler boundary search + attribution evaluation (Runner).")

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
    parser.add_argument(
        "--crawler_mode",
        type=str,
        default="random",
        choices=["random", "gradient"],
        help="Crawler mode: 'random' ring or 'gradient' projection",
    )

    # FGSM init parameters (used by crawler for initial boundary point)
    parser.add_argument("--fgsm_step_size", type=float, default=0.005, help="FGSM step size for crawler init")
    parser.add_argument("--fgsm_max_iters", type=int, default=100, help="FGSM max iterations for crawler init")

    # Visualization
    parser.add_argument("--vis_dir", type=str, default="results/eval/vis")
    parser.add_argument("--vis_points", type=int, default=200, help="How many per-point plots to save (2D only).")

    # MC sampling for sphere volume
    parser.add_argument("--sphere_samples", type=int, default=20000, help="MC samples for sphere volume calculation")

    # Attribution params (Runner supports vg/sg/ig/ks/lime)
    parser.add_argument(
        "--attr_method",
        type=str,
        default="ig",
        choices=["ig", "vg", "sg", "ks", "lime", "none"],
        help="Attribution method to compute: vg, sg, ig, ks, lime, or none.",
    )
    parser.add_argument("--ig_steps", type=int, default=64, help="Number of steps for Integrated Gradients.")
    parser.add_argument("--topk", type=int, default=10, help="Top-k features for Jaccard similarity (tabular).")

    # SmoothGrad knobs (optional)
    parser.add_argument("--sg_samples", type=int, default=50, help="NoiseTunnel SmoothGrad samples.")
    parser.add_argument(
        "--sg_nt_type",
        type=str,
        default="smoothgrad",
        choices=["smoothgrad", "smoothgrad_sq", "vargrad"],
        help="NoiseTunnel type.",
    )
    parser.add_argument("--sg_stdevs", type=float, default=0.1, help="SmoothGrad noise stdevs (absolute scale).")

    # KernelSHAP / LIME knobs (optional)
    parser.add_argument("--perturb_samples", type=int, default=300, help="n_samples for KernelSHAP/LIME.")

    # Metrics knobs
    parser.add_argument("--fidelity_steps", type=int, default=25, help="Steps for deletion/insertion curves.")
    parser.add_argument("--stability_eps", type=float, default=0.01, help="Perturb radius for stability.")
    parser.add_argument("--stability_samples", type=int, default=30, help="Perturb samples for stability.")

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # Setup paths
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load Data (includes GS oracle results saved in the CSV pipeline)
    logger.info(f"Loading data from {args.data_path}...")
    X, y, dataset_metadata, gs_success, gs_radius, x_enemy, x_boundary, dist_x0_enemy, dist_x0_boundary = load_dataset(
        csv_path=args.data_path
    )

    # Load Model
    logger.info(f"Loading model from {args.model_path} ({args.model_type})...")
    model = load_model(args.model_path, input_dim=X.shape[1], model_type=args.model_type, num_classes=args.num_classes)
    model.eval()

    device = next(model.parameters()).device

    # Setup crawler
    fgsm_params = {"step_size": args.fgsm_step_size, "max_steps": args.fgsm_max_iters, "clamp": None}
    crawl_params = {
        "mode": args.crawler_mode,
        "max_iterations": args.crawler_iter,
        "num_samples": args.crawler_samples,
        "step_size": args.crawler_step,
        "bisection_steps": 10,
    }
    crawler = BoundaryCrawler(model=model, device=device, fgsm_params=fgsm_params, crawl_params=crawl_params)

    # Setup attribution runner
    runner = AttributionRunner(model=model, device=device)

    def make_attr_cfg() -> Optional[AttributionConfig]:
        if args.attr_method == "none":
            return None

        cfg = AttributionConfig(method=args.attr_method)

        # Common
        cfg.abs_attributions = False
        cfg.internal_batch_size = None  # set if you want (e.g., 64 for ks/lime)

        # IG
        cfg.ig_steps = int(args.ig_steps)

        # SmoothGrad
        cfg.sg_n_samples = int(args.sg_samples)
        cfg.sg_stdevs = float(args.sg_stdevs)
        cfg.sg_nt_type = str(args.sg_nt_type)

        # KernelSHAP / LIME
        cfg.perturb_samples = int(args.perturb_samples)
        cfg.feature_mask = None  # set if you group features

        return cfg

    def run_search(x_input: torch.Tensor):
        x_np = x_input.detach().cpu().numpy()
        res = crawler.search(x_np)
        b_point = torch.from_numpy(np.asarray(res.x_boundary, dtype=np.float32)).to(device)
        b_point = b_point.squeeze(0) if b_point.ndim == 2 else b_point
        return b_point, res.success

    def predict_class(x_input: torch.Tensor) -> int:
        with torch.no_grad():
            logits = model(x_input.unsqueeze(0))
            return int(logits.argmax(dim=-1).item())

    logger.info("Running crawler boundary search + attribution evaluation...")
    rows: List[Dict[str, Any]] = []
    n = len(X)

    for i in tqdm(range(len(X))):
        x_np = X[i]
        x_input = torch.tensor(x_np, dtype=torch.float32, device=device)  # (d,)
        y_i = int(y[i])

        # GS result loaded from dataset
        gs_res = GSOracleResult(
            x_start=x_np,
            x_enemy=x_enemy[i],
            x_boundary=x_boundary[i],
            radius_found=float(gs_radius[i]),
            success=bool(gs_success[i]),
            meta={"method": "growing_spheres_oracle"},
        )

        # Crawler boundary search
        crawler_b_point, crawler_success = run_search(x_input)
        b_point_np = crawler_b_point.detach().cpu().numpy().astype(np.float32)
        crawler_dist = float(np.linalg.norm(b_point_np - x_np))

        # Angle metric
        if crawler_success and gs_res.success:
            angle_deg = angle_at_x_degrees(x_np, b_point_np, np.asarray(gs_res.x_boundary, dtype=np.float32))
        else:
            angle_deg = np.nan

        # Decision boundary sphere volume (red strike volume)
        if crawler_success:
            sphere_result = decision_boundary_sphere_volume(
                model, x_np, crawler_dist, device=device, n_samples=args.sphere_samples, seed=args.seed + i
            )
        else:
            sphere_result = {"red_frac": np.nan, "red_vol": np.nan}

        # Ring / annulus counts
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

        # ---------------- Attribution + metrics (Runner) ----------------
        cfg = make_attr_cfg()
        attr_target = predict_class(x_input)  # explain predicted class
        zero_base = torch.zeros_like(x_input)  # (d,)

        # baselines
        gs_base_t = None
        if gs_res.success:
            try:
                gs_base_t = _np_to_torch_float(gs_res.x_boundary, device=device).view(-1)  # (d,)
            except Exception:
                gs_base_t = None

        crawler_base_t = crawler_b_point if crawler_success else None

        # attributions (torch, (1,d))
        attr_zero = attr_crawler = attr_gs = None

        # defaults NaN
        attr_jacc_crawler_gs = np.nan
        attr_spearman_crawler_gs = np.nan
        attr_delta_zero_vs_crawler_l2 = np.nan
        attr_delta_zero_vs_gs_l2 = np.nan

        attr_fid_del_auc_gs = np.nan
        attr_fid_ins_auc_gs = np.nan
        attr_fid_del_delta_topk_gs = np.nan
        attr_fid_ins_delta_topk_gs = np.nan

        attr_stability_lipschitz_gs = np.nan
        attr_stability_mean_attr_change_gs = np.nan

        attr_cos_truegrad_gs = np.nan
        attr_align_gs = np.nan
        attr_align_crawler = np.nan

        if cfg is not None:
            # Zero baseline
            try:
                attr_zero = compute_attr(runner, x_input, zero_base, attr_target, cfg)
            except Exception:
                attr_zero = None

            # Crawler baseline
            if crawler_base_t is not None:
                try:
                    attr_crawler = compute_attr(runner, x_input, crawler_base_t, attr_target, cfg)
                except Exception:
                    attr_crawler = None

            # GS baseline
            if gs_base_t is not None:
                try:
                    attr_gs = compute_attr(runner, x_input, gs_base_t, attr_target, cfg)
                except Exception:
                    attr_gs = None

            # Disagreement: crawler vs GS
            if (attr_crawler is not None) and (attr_gs is not None):
                attr_jacc_crawler_gs = jaccard_topk(attr_crawler, attr_gs, k=args.topk)
                attr_spearman_crawler_gs = spearman_rank_corr(attr_crawler, attr_gs)

            # Baseline effect norms
            if (attr_zero is not None) and (attr_crawler is not None):
                attr_delta_zero_vs_crawler_l2 = attribution_delta_norm(attr_zero, attr_crawler, p=2)
            if (attr_zero is not None) and (attr_gs is not None):
                attr_delta_zero_vs_gs_l2 = attribution_delta_norm(attr_zero, attr_gs, p=2)

            # Fidelity / Stability / Ground truth / Alignment using GS baseline
            if (attr_gs is not None) and (gs_base_t is not None):
                try:
                    fid = fidelity_deletion_insertion(
                        model=model,
                        x=x_input,
                        attr=attr_gs,
                        baseline=gs_base_t,
                        target=attr_target,
                        steps=args.fidelity_steps,
                        topk=args.topk,
                        use_logits=True,
                    )
                    attr_fid_del_auc_gs = fid.deletion_auc
                    attr_fid_ins_auc_gs = fid.insertion_auc
                    attr_fid_del_delta_topk_gs = fid.deletion_delta_topk
                    attr_fid_ins_delta_topk_gs = fid.insertion_delta_topk
                except Exception:
                    pass

                try:
                    stab = stability_local_lipschitz(
                        attr_fn=lambda z: runner.attribute(
                            x=z if z.ndim == 2 else z.unsqueeze(0),
                            baseline=gs_base_t.unsqueeze(0),
                            target=attr_target,
                            cfg=cfg,
                        ),
                        x=x_input,
                        eps=args.stability_eps,
                        n_samples=args.stability_samples,
                        norm="l2",
                        seed=args.seed + i,
                    )
                    attr_stability_lipschitz_gs = stab.lipschitz_est
                    attr_stability_mean_attr_change_gs = stab.mean_attr_change
                except Exception as e:
                    pass

                try:
                    true_g = true_gradient_attribution(model, x_input, target=attr_target)
                    attr_cos_truegrad_gs = cosine_with_true_gradient(attr_gs, true_g)
                except Exception:
                    pass

                try:
                    attr_align_gs = boundary_alignment_cosine(attr_gs, x_input, gs_base_t)
                except Exception:
                    pass

            # Alignment with crawler baseline too
            if (attr_crawler is not None) and (crawler_base_t is not None):
                try:
                    attr_align_crawler = boundary_alignment_cosine(attr_crawler, x_input, crawler_base_t)
                except Exception:
                    pass

        # ---------------- Row ----------------
        row = {
            "idx": i,
            "y": y_i,
            "pred_target": int(attr_target),

            "crawler_success": bool(crawler_success),
            "gs_success": bool(gs_res.success),

            "gs_radius_found": float(gs_res.radius_found),
            "angle_x_crawler_gs_deg": float(angle_deg),

            "sphere_frac": float(sphere_result["red_frac"]),
            "sphere_vol": float(sphere_result["red_vol"]),

            "dist_x_to_crawler": l2(x_np, b_point_np) if crawler_success else np.nan,
            "dist_x_to_gs": l2(x_np, np.asarray(gs_res.x_boundary, dtype=np.float32)) if gs_res.success else np.nan,
            "dist_crawler_to_gs": (
                l2(b_point_np, np.asarray(gs_res.x_boundary, dtype=np.float32))
                if (crawler_success and gs_res.success) else np.nan
            ),

            "ring_n": int(ring.get("ring_n", 0)),
            "ring_r_min": float(ring.get("ring_r_min", np.nan)),
            "ring_r_max": float(ring.get("ring_r_max", np.nan)),
            "ring_count_y0": int(ring.get("ring_count_y0", 0)),
            "ring_count_y1": int(ring.get("ring_count_y1", 0)),
            "ring_count_pred0": int(ring.get("ring_count_pred0", 0)),
            "ring_count_pred1": int(ring.get("ring_count_pred1", 0)),

            # Attribution config
            "attr_method": str(cfg.method) if cfg is not None else "none",
            "attr_ig_steps": int(args.ig_steps),
            "attr_topk": int(args.topk),
            "attr_sg_samples": int(args.sg_samples),
            "attr_sg_stdevs": float(args.sg_stdevs),
            "attr_sg_nt_type": str(args.sg_nt_type),
            "attr_perturb_samples": int(args.perturb_samples),

            # Disagreement
            "attr_jacc_crawler_gs@k": float(attr_jacc_crawler_gs),
            "attr_spearman_crawler_gs": float(attr_spearman_crawler_gs),

            # Baseline effect
            "attr_delta_zero_vs_crawler_l2": float(attr_delta_zero_vs_crawler_l2),
            "attr_delta_zero_vs_gs_l2": float(attr_delta_zero_vs_gs_l2),

            # Fidelity (using GS baseline)
            "attr_fidelity_del_auc_gs": float(attr_fid_del_auc_gs),
            "attr_fidelity_ins_auc_gs": float(attr_fid_ins_auc_gs),
            "attr_fidelity_del_delta_topk_gs": float(attr_fid_del_delta_topk_gs),
            "attr_fidelity_ins_delta_topk_gs": float(attr_fid_ins_delta_topk_gs),

            # Stability (using GS baseline)
            "attr_stability_lipschitz_gs": float(attr_stability_lipschitz_gs),
            "attr_stability_mean_attr_change_gs": float(attr_stability_mean_attr_change_gs),

            # Synthetic ground truth proxy
            "attr_cosine_truegrad_gs": float(attr_cos_truegrad_gs),

            # Boundary alignment
            "attr_boundary_alignment_gs": float(attr_align_gs),
            "attr_boundary_alignment_crawler": float(attr_align_crawler),
        }
        rows.append(row)

        if i % 25 == 0:
            logger.info("Processed %d/%d points", i + 1, n)

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
                b_crawler=b_point_np if row["crawler_success"] else None,
                b_gs=np.asarray(gs_res.x_boundary, dtype=np.float32) if row["gs_success"] else None,
                gs_radius=row["gs_radius_found"] if row["gs_success"] else None,
                save_path=out_path,
                X_train=X,
                y_train=y,
                device=device,
                title=title,
                crawler_circle=True,
            )

    df = pd.DataFrame(rows)
    report_path = save_dir / "report.csv"
    df.to_csv(report_path, index=False)

    logger.info("Saved report CSV: %s", report_path)
    logger.info("Summary:\n%s", df.describe(include="all"))


if __name__ == "__main__":
    main()

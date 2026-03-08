from __future__ import annotations

import argparse
import logging
from itertools import combinations
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

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

try:
    from captum.metrics import sensitivity_max
except Exception:  # pragma: no cover - optional dependency
    sensitivity_max = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_ATTR_METHODS = ("ig", "lime", "ks")
DEFAULT_BASELINES = ("zero", "boundarycrawler", "gs")


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _np_to_torch_float(x, device) -> torch.Tensor:
    arr = np.asarray(x)
    if arr.dtype == object:
        arr = np.asarray(arr.tolist(), dtype=np.float32)
    else:
        arr = arr.astype(np.float32, copy=False)
    return torch.tensor(arr, dtype=torch.float32, device=device)


def compute_attr(
    runner: AttributionRunner,
    x: torch.Tensor,
    baseline: torch.Tensor,
    target: int,
    cfg: AttributionConfig,
) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if baseline.ndim == 1:
        baseline = baseline.unsqueeze(0)
    attr = runner.attribute(x=x, baseline=baseline, target=target, cfg=cfg)
    return attr.detach()


def build_attr_cfg(method: str, args: argparse.Namespace) -> AttributionConfig:
    cfg = AttributionConfig(method=method)
    cfg.abs_attributions = False
    cfg.internal_batch_size = None
    cfg.ig_steps = int(args.ig_steps)
    cfg.sg_n_samples = int(args.sg_samples)
    cfg.sg_stdevs = float(args.sg_stdevs)
    cfg.sg_nt_type = str(args.sg_nt_type)
    cfg.perturb_samples = int(args.perturb_samples)
    cfg.perturbations_per_eval = int(args.perturbations_per_eval)
    cfg.feature_mask = None
    return cfg


def parse_csv_arg(raw: str, allowed: Tuple[str, ...]) -> List[str]:
    items = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not items:
        return list(allowed)
    invalid = sorted(set(items) - set(allowed))
    if invalid:
        raise ValueError(f"Unsupported values: {invalid}. Allowed: {allowed}")
    return items


def format_metric_name(method: str, baseline: str, metric: str) -> str:
    return f"attr_{method}_{baseline}_{metric}"


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def compute_captum_sensitivity(
    runner: AttributionRunner,
    cfg: AttributionConfig,
    x_input: torch.Tensor,
    baseline: torch.Tensor,
    target: int,
    seed: int,
) -> float:
    if sensitivity_max is None:
        return np.nan

    x_batched = x_input.unsqueeze(0) if x_input.ndim == 1 else x_input
    baseline_batched = baseline.unsqueeze(0) if baseline.ndim == 1 else baseline

    torch.manual_seed(seed)
    np.random.seed(seed)

    def explain_func(inputs: torch.Tensor) -> torch.Tensor:
        return runner.attribute(
            x=inputs,
            baseline=baseline_batched.expand(inputs.shape[0], -1),
            target=target,
            cfg=cfg,
        )

    try:
        val = sensitivity_max(
            explain_func,
            x_batched,
            perturb_radius=float(0.02),
            n_perturb_samples=10,
            norm_ord="fro",
            max_examples_per_batch=None,
        )
        if isinstance(val, torch.Tensor):
            return safe_float(val.detach().mean().item())
        return safe_float(val)
    except Exception:
        return np.nan


def main():
    parser = argparse.ArgumentParser(
        description="Run boundary search + attribution evaluation for multiple methods/baselines."
    )

    parser.add_argument("--data_path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "linear"], help="Type of model")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of output classes")
    parser.add_argument("--save_dir", type=str, default="results/boundary_search", help="Directory to save results")

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

    parser.add_argument("--fgsm_step_size", type=float, default=0.005, help="FGSM step size for crawler init")
    parser.add_argument("--fgsm_max_iters", type=int, default=100, help="FGSM max iterations for crawler init")

    parser.add_argument("--vis_dir", type=str, default="results/eval/vis")
    parser.add_argument("--vis_points", type=int, default=200, help="How many per-point plots to save (2D only).")

    parser.add_argument("--sphere_samples", type=int, default=20000, help="MC samples for sphere volume calculation")

    parser.add_argument(
        "--attr_methods",
        type=str,
        default=",".join(DEFAULT_ATTR_METHODS),
        help="Comma-separated attribution methods to run. Supported: ig,lime,ks",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default=",".join(DEFAULT_BASELINES),
        help="Comma-separated baselines to run. Supported: zero,boundarycrawler,gs",
    )
    parser.add_argument("--ig_steps", type=int, default=64, help="Number of steps for Integrated Gradients.")
    parser.add_argument("--topk", type=int, default=10, help="Top-k features for Jaccard similarity.")

    parser.add_argument("--sg_samples", type=int, default=50, help="NoiseTunnel SmoothGrad samples.")
    parser.add_argument(
        "--sg_nt_type",
        type=str,
        default="smoothgrad",
        choices=["smoothgrad", "smoothgrad_sq", "vargrad"],
        help="NoiseTunnel type.",
    )
    parser.add_argument("--sg_stdevs", type=float, default=0.1, help="SmoothGrad noise stdevs.")
    parser.add_argument("--perturb_samples", type=int, default=500, help="n_samples for KernelSHAP/LIME.")
    parser.add_argument("--perturbations_per_eval", type=int, default=64, help="Batch size for KernelSHAP/LIME.")

    parser.add_argument("--fidelity_steps", type=int, default=25, help="Steps for deletion/insertion curves.")
    parser.add_argument("--stability_eps", type=float, default=0.01, help="Perturb radius for stability.")
    parser.add_argument("--stability_samples", type=int, default=30, help="Perturb samples for stability.")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    attr_methods = parse_csv_arg(args.attr_methods, DEFAULT_ATTR_METHODS)
    baseline_names = parse_csv_arg(args.baselines, DEFAULT_BASELINES)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data from %s...", args.data_path)
    X, y, dataset_metadata, gs_success, gs_radius, x_enemy, x_boundary, dist_x0_enemy, dist_x0_boundary = load_dataset(
        csv_path=args.data_path
    )
    dataset_name = dataset_metadata.get("dataset_name", Path(args.data_path).stem) if isinstance(dataset_metadata, dict) else Path(args.data_path).stem

    logger.info("Loading model from %s (%s)...", args.model_path, args.model_type)
    model = load_model(args.model_path, input_dim=X.shape[1], model_type=args.model_type, num_classes=args.num_classes)
    model.eval()
    device = next(model.parameters()).device

    fgsm_params = {"step_size": args.fgsm_step_size, "max_steps": args.fgsm_max_iters, "clamp": None}
    crawl_params = {
        "mode": args.crawler_mode,
        "max_iterations": args.crawler_iter,
        "num_samples": args.crawler_samples,
        "step_size": args.crawler_step,
        "bisection_steps": 10,
    }
    crawler = BoundaryCrawler(model=model, device=device, fgsm_params=fgsm_params, crawl_params=crawl_params)
    runner = AttributionRunner(model=model, device=device)
    cfg_map = {method: build_attr_cfg(method, args) for method in attr_methods}

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
    pairwise_rows: List[Dict[str, Any]] = []
    n = len(X)

    for i in tqdm(range(n)):
        x_np = X[i]
        x_input = torch.tensor(x_np, dtype=torch.float32, device=device)
        y_i = int(y[i])

        gs_res = GSOracleResult(
            x_start=x_np,
            x_enemy=x_enemy[i],
            x_boundary=x_boundary[i],
            radius_found=float(dist_x0_boundary[i]),
            success=bool(gs_success[i]),
            meta={"method": "growing_spheres_oracle"},
        )

        crawler_b_point, crawler_success = run_search(x_input)
        b_point_np = crawler_b_point.detach().cpu().numpy().astype(np.float32)
        crawler_dist = float(np.linalg.norm(b_point_np - x_np))

        if crawler_success and gs_res.success:
            angle_deg = angle_at_x_degrees(x_np, b_point_np, np.asarray(gs_res.x_boundary, dtype=np.float32))
        else:
            angle_deg = np.nan

        if crawler_success:
            sphere_result = decision_boundary_sphere_volume(
                model, x_np, crawler_dist, device=device, n_samples=args.sphere_samples, seed=args.seed + i
            )
        else:
            sphere_result = {"red_frac": -1, "red_vol": -1}

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

        attr_target = predict_class(x_input)

        zero_base = torch.zeros_like(x_input)
        boundarycrawler_base = crawler_b_point if crawler_success else None
        gs_base = None
        if gs_res.success and gs_res.x_boundary is not None:
            gs_base = _np_to_torch_float(gs_res.x_boundary, device=device).squeeze(0)

        baselines: Dict[str, Optional[torch.Tensor]] = {
            "zero": zero_base,
            "boundarycrawler": boundarycrawler_base,
            "gs": gs_base,
        }

        attr_results: Dict[str, Dict[str, Any]] = {}
        true_g = None
        try:
            true_g = true_gradient_attribution(model, x_input, target=attr_target)
        except Exception:
            true_g = None

        for method in attr_methods:
            cfg = cfg_map[method]
            for baseline_name in baseline_names:
                baseline = baselines.get(baseline_name)
                key = f"{method}_{baseline_name}"
                metrics = {
                    "success": False,
                    "attr": None,
                    "sensitivity": np.nan,
                    "fidelity_del_auc": np.nan,
                    "fidelity_ins_auc": np.nan,
                    "fidelity_del_delta_topk": np.nan,
                    "fidelity_ins_delta_topk": np.nan,
                    "stability_lipschitz": np.nan,
                    "stability_mean_attr_change": np.nan,
                    "cosine_truegrad": np.nan,
                    "boundary_alignment": np.nan,
                }

                if baseline is None:
                    attr_results[key] = metrics
                    continue

                attr = compute_attr(runner, x_input, baseline, attr_target, cfg)
                metrics["attr"] = attr
                metrics["success"] = True

                try:
                    metrics["sensitivity"] = compute_captum_sensitivity(
                        runner=runner,
                        cfg=cfg,
                        x_input=x_input,
                        baseline=baseline,
                        target=attr_target,
                        seed=args.seed + i,
                    )
                except Exception:
                    pass

                try:
                    fid = fidelity_deletion_insertion(
                        model=model,
                        x=x_input,
                        attr=metrics["attr"],
                        baseline=baseline,
                        target=attr_target,
                        steps=args.fidelity_steps,
                        topk=args.topk,
                        use_logits=True,
                    )
                    metrics["fidelity_del_auc"] = safe_float(fid.deletion_auc)
                    metrics["fidelity_ins_auc"] = safe_float(fid.insertion_auc)
                    metrics["fidelity_del_delta_topk"] = safe_float(fid.deletion_delta_topk)
                    metrics["fidelity_ins_delta_topk"] = safe_float(fid.insertion_delta_topk)
                except Exception:
                    pass

                try:
                    baseline_batched = baseline.unsqueeze(0) if baseline.ndim == 1 else baseline
                    stab = stability_local_lipschitz(
                        attr_fn=lambda z, _cfg=cfg, _baseline=baseline_batched: runner.attribute(
                            x=z if z.ndim == 2 else z.unsqueeze(0),
                            baseline=_baseline.expand((z.shape[0] if z.ndim == 2 else 1), -1),
                            target=attr_target,
                            cfg=_cfg,
                        ),
                        x=x_input,
                        eps=args.stability_eps,
                        n_samples=args.stability_samples,
                        norm="l2",
                        seed=args.seed + i,
                    )
                    metrics["stability_lipschitz"] = safe_float(stab.lipschitz_est)
                    metrics["stability_mean_attr_change"] = safe_float(stab.mean_attr_change)
                except Exception:
                    pass

                if true_g is not None:
                    try:
                        metrics["cosine_truegrad"] = safe_float(cosine_with_true_gradient(metrics["attr"], true_g))
                    except Exception:
                        pass

                try:
                    metrics["boundary_alignment"] = safe_float(
                        boundary_alignment_cosine(metrics["attr"], x_input, baseline)
                    )
                except Exception:
                    pass

                attr_results[key] = metrics

        row: Dict[str, Any] = {
            "idx": i,
            "dataset": dataset_name,
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
                l2(b_point_np, np.asarray(gs_res.x_boundary, dtype=np.float32)) if (crawler_success and gs_res.success) else np.nan
            ),
            "ring_n": int(ring.get("ring_n", 0)),
            "ring_r_min": float(ring.get("ring_r_min", np.nan)),
            "ring_r_max": float(ring.get("ring_r_max", np.nan)),
            "ring_count_y0": int(ring.get("ring_count_y0", 0)),
            "ring_count_y1": int(ring.get("ring_count_y1", 0)),
            "ring_count_pred0": int(ring.get("ring_count_pred0", 0)),
            "ring_count_pred1": int(ring.get("ring_count_pred1", 0)),
            "attr_methods": ",".join(attr_methods),
            "attr_baselines": ",".join(baseline_names),
            "attr_ig_steps": int(args.ig_steps),
            "attr_topk": int(args.topk),
            "attr_sg_samples": int(args.sg_samples),
            "attr_sg_stdevs": float(args.sg_stdevs),
            "attr_sg_nt_type": str(args.sg_nt_type),
            "attr_perturb_samples": int(args.perturb_samples),
            "attr_perturbations_per_eval": int(args.perturbations_per_eval),
        }

        for key, metrics in attr_results.items():
            method, baseline_name = key.split("_", 1)
            row[format_metric_name(method, baseline_name, "success")] = bool(metrics["success"])
            row[format_metric_name(method, baseline_name, "sensitivity")] = safe_float(metrics["sensitivity"])
            row[format_metric_name(method, baseline_name, "fidelity_del_auc")] = safe_float(metrics["fidelity_del_auc"])
            row[format_metric_name(method, baseline_name, "fidelity_ins_auc")] = safe_float(metrics["fidelity_ins_auc"])
            row[format_metric_name(method, baseline_name, "fidelity_del_delta_topk")] = safe_float(metrics["fidelity_del_delta_topk"])
            row[format_metric_name(method, baseline_name, "fidelity_ins_delta_topk")] = safe_float(metrics["fidelity_ins_delta_topk"])
            row[format_metric_name(method, baseline_name, "stability_lipschitz")] = safe_float(metrics["stability_lipschitz"])
            row[format_metric_name(method, baseline_name, "stability_mean_attr_change")] = safe_float(metrics["stability_mean_attr_change"])
            row[format_metric_name(method, baseline_name, "cosine_truegrad")] = safe_float(metrics["cosine_truegrad"])
            row[format_metric_name(method, baseline_name, "boundary_alignment")] = safe_float(metrics["boundary_alignment"])

        for (key_a, res_a), (key_b, res_b) in combinations(attr_results.items(), 2):
            if (res_a["attr"] is None) or (res_b["attr"] is None):
                continue
            try:
                jacc = safe_float(jaccard_topk(res_a["attr"], res_b["attr"], k=args.topk))
            except Exception:
                jacc = np.nan
            try:
                spear = safe_float(spearman_rank_corr(res_a["attr"], res_b["attr"]))
            except Exception:
                spear = np.nan
            try:
                delta_l2 = safe_float(attribution_delta_norm(res_a["attr"], res_b["attr"], p=2))
            except Exception:
                delta_l2 = np.nan

            pairwise_rows.append(
                {
                    "idx": i,
                    "dataset": dataset_name,
                    "method_a": key_a,
                    "method_b": key_b,
                    "jaccard_topk": jacc,
                    "spearman_rank": spear,
                    "delta_l2": delta_l2,
                }
            )

        rows.append(row)

        if i % 25 == 0:
            logger.info("Processed %d/%d points", i + 1, n)

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

    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_path = save_dir / "report_pairwise_disagreement.csv"
    pairwise_df.to_csv(pairwise_path, index=False)

    summary_rows: List[Dict[str, Any]] = []
    for method in attr_methods:
        for baseline_name in baseline_names:
            prefix = f"attr_{method}_{baseline_name}_"
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "method": method,
                    "baseline": baseline_name,
                    "n": int(len(df)),
                    "success_rate": safe_float(df.get(prefix + "success", pd.Series(dtype=float)).mean()),
                    "sensitivity_mean": safe_float(df.get(prefix + "sensitivity", pd.Series(dtype=float)).mean()),
                    "stability_lipschitz_mean": safe_float(df.get(prefix + "stability_lipschitz", pd.Series(dtype=float)).mean()),
                    "stability_mean_attr_change_mean": safe_float(df.get(prefix + "stability_mean_attr_change", pd.Series(dtype=float)).mean()),
                    "fidelity_del_auc_mean": safe_float(df.get(prefix + "fidelity_del_auc", pd.Series(dtype=float)).mean()),
                    "fidelity_ins_auc_mean": safe_float(df.get(prefix + "fidelity_ins_auc", pd.Series(dtype=float)).mean()),
                    "fidelity_del_delta_topk_mean": safe_float(df.get(prefix + "fidelity_del_delta_topk", pd.Series(dtype=float)).mean()),
                    "fidelity_ins_delta_topk_mean": safe_float(df.get(prefix + "fidelity_ins_delta_topk", pd.Series(dtype=float)).mean()),
                    "cosine_truegrad_mean": safe_float(df.get(prefix + "cosine_truegrad", pd.Series(dtype=float)).mean()),
                    "boundary_alignment_mean": safe_float(df.get(prefix + "boundary_alignment", pd.Series(dtype=float)).mean()),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = save_dir / "report_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if not pairwise_df.empty:
        pairwise_summary_df = (
            pairwise_df.groupby(["dataset", "method_a", "method_b"], dropna=False)[["jaccard_topk", "spearman_rank", "delta_l2"]]
            .mean()
            .reset_index()
        )
    else:
        pairwise_summary_df = pd.DataFrame(
            columns=["dataset", "method_a", "method_b", "jaccard_topk", "spearman_rank", "delta_l2"]
        )
    pairwise_summary_path = save_dir / "report_pairwise_summary.csv"
    pairwise_summary_df.to_csv(pairwise_summary_path, index=False)

    logger.info("Saved report CSV: %s", report_path)
    logger.info("Saved pairwise disagreement CSV: %s", pairwise_path)
    logger.info("Saved summary CSV: %s", summary_path)
    logger.info("Saved pairwise summary CSV: %s", pairwise_summary_path)
    logger.info("Summary:\n%s", summary_df)


if __name__ == "__main__":
    main()
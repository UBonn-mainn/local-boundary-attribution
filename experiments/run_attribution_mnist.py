from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from attribution.config import AttributionConfig
from attribution.runner import AttributionRunner
from boundary_search.boundary_walker import BoundaryCrawler
from boundary_search.growing_spheres_oracle import GSOracleResult
from evaluation.angle_metrics import angle_at_x_degrees
from evaluation.attribution_metrics.baseline_effect import attribution_delta_norm
from evaluation.attribution_metrics.boundary_alignment import boundary_alignment_cosine
from evaluation.attribution_metrics.disagreement import jaccard_topk, spearman_rank_corr
from evaluation.attribution_metrics.fidelity import fidelity_deletion_insertion
from evaluation.attribution_metrics.ground_truth import cosine_with_true_gradient, true_gradient_attribution
from evaluation.attribution_metrics.stability import stability_local_lipschitz
from evaluation.decision_boundary_sphere_volume import decision_boundary_sphere_volume
from utils.data.dataset_utils import get_mnist_dataloaders
from utils.entities.small_mnist_cnn import SmallMNISTCNN

try:
    from captum.metrics import sensitivity_max
except Exception:
    sensitivity_max = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_ATTR_METHODS = ("ig", "lime", "ks")
DEFAULT_BASELINES = ("zero", "boundarycrawler", "gs")


# ============================================================
# Generic helpers
# ============================================================

def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


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


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.reshape(-1) - b.reshape(-1)))


def _np_to_torch_image(x: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        side = int(np.sqrt(arr.size))
        if side * side != arr.size:
            raise ValueError(f"Cannot reshape flat input of size {arr.size} into square image.")
        arr = arr.reshape(1, side, side)
    elif arr.ndim == 2:
        arr = arr.reshape(1, arr.shape[0], arr.shape[1])
    elif arr.ndim == 3:
        pass
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")
    return torch.tensor(arr, dtype=torch.float32, device=device)


@torch.no_grad()
def predict_class(model: torch.nn.Module, x_input: torch.Tensor) -> int:
    x_b = x_input.unsqueeze(0) if x_input.ndim == 3 else x_input
    logits = model(x_b)
    return int(logits.argmax(dim=-1).item())


# ============================================================
# GS CSV loading
# ============================================================

def load_gs_results(csv_path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gs_success = str(row["gs_success"]).lower() == "true"
            rows.append(
                {
                    "sample_idx": int(row["sample_idx"]),
                    "num_features": int(row["num_features"]),
                    "x0": np.array(ast.literal_eval(row["x0"]), dtype=np.float32),
                    "class": int(row["class"]),
                    "gs_success": gs_success,
                    "found_radius": float(row["found_radius"]) if row["found_radius"] != "" else np.nan,
                    "x_enemy": np.array(ast.literal_eval(row["x_enemy"]), dtype=np.float32) if row["x_enemy"] != "" else None,
                    "x_boundary": np.array(ast.literal_eval(row["x_boundary"]), dtype=np.float32) if row["x_boundary"] != "" else None,
                    "dist_x0_enemy": float(row["dist_x0_enemy"]) if row["dist_x0_enemy"] != "" else np.nan,
                    "dist_x0_boundary": float(row["dist_x0_boundary"]) if row["dist_x0_boundary"] != "" else np.nan,
                }
            )
    return rows


# ============================================================
# Attribution helpers
# ============================================================

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


def compute_attr(
    runner: AttributionRunner,
    x: torch.Tensor,
    baseline: torch.Tensor,
    target: int,
    cfg: AttributionConfig,
) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if baseline.ndim == 3:
        baseline = baseline.unsqueeze(0)
    attr = runner.attribute(x=x, baseline=baseline, target=target, cfg=cfg)
    return attr.detach()


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

    x_batched = x_input.unsqueeze(0) if x_input.ndim == 3 else x_input
    baseline_batched = baseline.unsqueeze(0) if baseline.ndim == 3 else baseline

    torch.manual_seed(seed)
    np.random.seed(seed)

    def explain_func(inputs: torch.Tensor) -> torch.Tensor:
        base = baseline_batched.expand(inputs.shape[0], -1, -1, -1)
        return runner.attribute(
            x=inputs,
            baseline=base,
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


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run attribution evaluation for MNIST with GS and BoundaryCrawler baselines.")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--gs_csv", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results/mnist_eval")

    parser.add_argument("--digit_a", type=int, default=None)
    parser.add_argument("--digit_b", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=None)

    parser.add_argument("--crawler_samples", type=int, default=10)
    parser.add_argument("--crawler_iter", type=int, default=50)
    parser.add_argument("--crawler_step", type=float, default=0.1)
    parser.add_argument("--crawler_mode", type=str, default="random", choices=["random", "gradient"])
    parser.add_argument("--fgsm_step_size", type=float, default=0.005)
    parser.add_argument("--fgsm_max_iters", type=int, default=100)

    parser.add_argument("--sphere_samples", type=int, default=20000)

    parser.add_argument("--attr_methods", type=str, default=",".join(DEFAULT_ATTR_METHODS))
    parser.add_argument("--baselines", type=str, default=",".join(DEFAULT_BASELINES))
    parser.add_argument("--ig_steps", type=int, default=64)
    parser.add_argument("--topk", type=int, default=50)

    parser.add_argument("--sg_samples", type=int, default=50)
    parser.add_argument("--sg_nt_type", type=str, default="smoothgrad", choices=["smoothgrad", "smoothgrad_sq", "vargrad"])
    parser.add_argument("--sg_stdevs", type=float, default=0.1)
    parser.add_argument("--perturb_samples", type=int, default=500)
    parser.add_argument("--perturbations_per_eval", type=int, default=64)

    parser.add_argument("--fidelity_steps", type=int, default=25)
    parser.add_argument("--stability_eps", type=float, default=0.01)
    parser.add_argument("--stability_samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    attr_methods = parse_csv_arg(args.attr_methods, DEFAULT_ATTR_METHODS)
    baseline_names = parse_csv_arg(args.baselines, DEFAULT_BASELINES)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)



    model_dir = Path(args.model_dir)
    with open(model_dir / "train_config.json", "r", encoding="utf-8") as f:
        train_cfg = json.load(f)

    binary_digits = None
    if args.digit_a is not None and args.digit_b is not None:
        binary_digits = (args.digit_a, args.digit_b)
    elif train_cfg.get("binary_digits") is not None:
        binary_digits = tuple(train_cfg["binary_digits"])

    logger.info("Loading MNIST data...")
    _, test_loader, inferred_num_classes = get_mnist_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        binary_digits=binary_digits,
    )

    gs_rows = load_gs_results(args.gs_csv)
    if len(gs_rows) == 0:
        raise ValueError(f"No GS rows loaded from {args.gs_csv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SmallMNISTCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))
    model.eval()

    fgsm_params = {"step_size": args.fgsm_step_size, "max_steps": args.fgsm_max_iters, "clamp": (0.0, 1.0)}
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
        x_np = x_input.detach().cpu().numpy().reshape(-1)
        res = crawler.search(x_np)
        b_point = np.asarray(res.x_boundary, dtype=np.float32)
        b_point_t = _np_to_torch_image(b_point, device=device)
        return b_point_t, bool(res.success), res

    logger.info("Running crawler boundary search + attribution evaluation on MNIST...")
    rows: List[Dict[str, Any]] = []
    pairwise_rows: List[Dict[str, Any]] = []

    for i, gs_row in enumerate(tqdm(gs_rows, total=len(gs_rows))):
        x_img = gs_row["x0"]
        y_i = gs_row["class"]

        x_input = _np_to_torch_image(x_img, device=device)
        x_np_flat = x_input.detach().cpu().numpy().reshape(-1)

        gs_res = GSOracleResult(
            x_start=gs_row["x0"],
            x_enemy=gs_row["x_enemy"] if gs_row["x_enemy"] is not None else gs_row["x0"],
            x_boundary=gs_row["x_boundary"] if gs_row["x_boundary"] is not None else gs_row["x0"],
            radius_found=float(gs_row["found_radius"]) if np.isfinite(gs_row["found_radius"]) else float("inf"),
            success=bool(gs_row["gs_success"]),
            meta={"method": "growing_spheres_oracle"},
        )

        crawler_b_point, crawler_success, crawler_res = run_search(x_input)
        b_point_np = crawler_b_point.detach().cpu().numpy().astype(np.float32)

        dist_x_to_crawler = (
            float(np.linalg.norm(b_point_np.reshape(-1) - x_np_flat))
            if crawler_success else np.nan
        )
        dist_x_to_gs = (
            float(np.linalg.norm(np.asarray(gs_res.x_boundary, dtype=np.float32).reshape(-1) - x_np_flat))
            if gs_res.success else np.nan
        )
        dist_crawler_to_gs = (
            float(np.linalg.norm(b_point_np.reshape(-1) - np.asarray(gs_res.x_boundary, dtype=np.float32).reshape(-1)))
            if (crawler_success and gs_res.success) else np.nan
        )
        angle_deg = (
            angle_at_x_degrees(
                x_np_flat,
                b_point_np.reshape(-1),
                np.asarray(gs_res.x_boundary, dtype=np.float32).reshape(-1),
            )
            if (crawler_success and gs_res.success) else np.nan
        )
        sphere_result = (
            decision_boundary_sphere_volume(
                model=model,
                x=x_np_flat,
                r=dist_x_to_crawler,
                device=device,
                n_samples=args.sphere_samples,
                seed=args.seed + i,
            )
            if (crawler_success and np.isfinite(dist_x_to_crawler) and dist_x_to_crawler > 0)
            else {"red_frac": np.nan, "red_vol": np.nan}
        )

        crawler_num_steps = int(crawler_res.num_steps) if hasattr(crawler_res, "num_steps") else np.nan
        gs_radius_found = safe_float(gs_res.radius_found) if gs_res.success else np.nan
        relative_distance_gap = (
            abs(dist_x_to_crawler - dist_x_to_gs) / max(dist_x_to_gs, 1e-12)
            if (crawler_success and gs_res.success and np.isfinite(dist_x_to_gs)) else np.nan
        )

        attr_target = predict_class(model, x_input)

        zero_base = torch.zeros_like(x_input)
        boundarycrawler_base = crawler_b_point if crawler_success else None
        gs_base = _np_to_torch_image(gs_res.x_boundary, device=device) if gs_res.success else None

        baselines: Dict[str, Optional[torch.Tensor]] = {
            "zero": zero_base,
            "boundarycrawler": boundarycrawler_base,
            "gs": gs_base,
        }

        attr_results: Dict[str, Dict[str, Any]] = {}

        true_g = None
        true_g = true_gradient_attribution(model, x_input, target=attr_target)
        # except Exception:
        #     true_g = None

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

                try:
                    attr = compute_attr(runner, x_input, baseline, attr_target, cfg)
                    metrics["attr"] = attr
                    metrics["success"] = True
                except Exception as e:
                    logger.warning("Attribution failed for idx=%d, %s: %s", i, key, e)
                    attr_results[key] = metrics
                    continue

                # try:
                metrics["sensitivity"] = compute_captum_sensitivity(
                    runner=runner,
                    cfg=cfg,
                    x_input=x_input,
                    baseline=baseline,
                    target=attr_target,
                    seed=args.seed + i,
                )
                # except Exception:
                #     pass

                # try:
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
                # except Exception:
                #     pass

                # try:
                baseline_batched = baseline.unsqueeze(0) if baseline.ndim == 3 else baseline
                stab = stability_local_lipschitz(
                    attr_fn=lambda z, _cfg=cfg, _baseline=baseline_batched: runner.attribute(
                        x=z if z.ndim == 4 else z.unsqueeze(0),
                        baseline=_baseline.expand((z.shape[0] if z.ndim == 4 else 1), -1, -1, -1),
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
                # except Exception:
                #     pass

                if true_g is not None:
                    # try:
                    metrics["cosine_truegrad"] = safe_float(cosine_with_true_gradient(metrics["attr"], true_g))
                    # except Exception:
                    #     pass

                # try:
                metrics["boundary_alignment"] = safe_float(
                    boundary_alignment_cosine(metrics["attr"], x_input, baseline)
                )
                # except Exception:
                #     pass

                attr_results[key] = metrics

        row: Dict[str, Any] = {
            "idx": i,
            "dataset": "mnist" if binary_digits is None else f"mnist_{binary_digits[0]}v{binary_digits[1]}",
            "y": int(y_i),
            "pred_target": int(attr_target),
            "crawler_success": bool(crawler_success),
            "gs_success": bool(gs_res.success),
            "crawler_num_steps": crawler_num_steps,
            "gs_radius_found": gs_radius_found,
            "dist_x_to_crawler": dist_x_to_crawler,
            "dist_x_to_gs": dist_x_to_gs,
            "dist_crawler_to_gs": dist_crawler_to_gs,
            "angle_x_crawler_gs_deg": angle_deg,
            "sphere_frac": safe_float(sphere_result["red_frac"]),
            "sphere_vol": safe_float(sphere_result["red_vol"]),
            "relative_distance_gap": relative_distance_gap,
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

            # try:
            jacc = safe_float(jaccard_topk(res_a["attr"], res_b["attr"], k=args.topk))
            # except Exception:
            #     jacc = np.nan

            # try:
            spear = safe_float(spearman_rank_corr(res_a["attr"], res_b["attr"]))
            # except Exception:
            #     spear = np.nan

            # try:
            delta_l2 = safe_float(attribution_delta_norm(res_a["attr"], res_b["attr"], p=2))
            # except Exception:
            #     delta_l2 = np.nan

            pairwise_rows.append(
                {
                    "idx": i,
                    "dataset": row["dataset"],
                    "method_a": key_a,
                    "method_b": key_b,
                    "jaccard_topk": jacc,
                    "spearman_rank": spear,
                    "delta_l2": delta_l2,
                }
            )

        rows.append(row)

        if i % 25 == 0:
            logger.info("Processed %d/%d points", i + 1, len(gs_rows))

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
                    "dataset": df["dataset"].iloc[0] if not df.empty else "mnist",
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

    boundary_summary = pd.DataFrame(
        [{
            "dataset": df["dataset"].iloc[0] if not df.empty else "mnist",
            "n": int(len(df)),
            "crawler_success_rate": safe_float(df["crawler_success"].mean()) if "crawler_success" in df else np.nan,
            "gs_success_rate": safe_float(df["gs_success"].mean()) if "gs_success" in df else np.nan,
            "crawler_num_steps_mean": safe_float(df["crawler_num_steps"].mean()) if "crawler_num_steps" in df else np.nan,
            "gs_radius_found_mean": safe_float(df["gs_radius_found"].mean()) if "gs_radius_found" in df else np.nan,
            "dist_x_to_crawler_mean": safe_float(df["dist_x_to_crawler"].mean()) if "dist_x_to_crawler" in df else np.nan,
            "dist_x_to_gs_mean": safe_float(df["dist_x_to_gs"].mean()) if "dist_x_to_gs" in df else np.nan,
            "dist_crawler_to_gs_mean": safe_float(df["dist_crawler_to_gs"].mean()) if "dist_crawler_to_gs" in df else np.nan,
            "angle_x_crawler_gs_deg_mean": safe_float(df["angle_x_crawler_gs_deg"].mean()) if "angle_x_crawler_gs_deg" in df else np.nan,
            "sphere_frac_mean": safe_float(df["sphere_frac"].mean()) if "sphere_frac" in df else np.nan,
            "sphere_vol_mean": safe_float(df["sphere_vol"].mean()) if "sphere_vol" in df else np.nan,
            "relative_distance_gap_mean": safe_float(df["relative_distance_gap"].mean()) if "relative_distance_gap" in df else np.nan,
        }]
    )
    boundary_summary_path = save_dir / "report_boundary_summary.csv"
    boundary_summary.to_csv(boundary_summary_path, index=False)

    logger.info("Saved report CSV: %s", report_path)
    logger.info("Saved pairwise disagreement CSV: %s", pairwise_path)
    logger.info("Saved summary CSV: %s", summary_path)
    logger.info("Saved pairwise summary CSV: %s", pairwise_summary_path)
    logger.info("Saved boundary summary CSV: %s", boundary_summary_path)
    logger.info("Summary:\n%s", summary_df)
    logger.info("Boundary summary:\n%s", boundary_summary)


if __name__ == "__main__":
    main()
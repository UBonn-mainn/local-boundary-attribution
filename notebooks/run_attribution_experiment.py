#!/usr/bin/env python
"""
Run Attribution Evaluation over Synthetic Experiments

This script traverses results/synthetic_experiments/* folders, infers:
- dimension (2d/8d/15d)
- dataset_type (moons/spirals/linear_blobs/3class/10class)
- num_classes

Then runs:
  experiments/run_attribution_method.py
(or whatever your attribution evaluation script is named)
with attribution enabled (AttributionRunner).

Expected per-experiment folder contains:
- data_wgs.csv
- model checkpoint: mlp_model.pth OR mlp_model.pth OR model.pth OR mlp_model.pth
- config.json (optional)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from utils.common.file_utils import collect_synthetic_experiments_map

# Fix OpenMP duplicate library error (MKL + PyTorch)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------
# Helpers
# -----------------------------

def find_data_csv(exp_dir: Path) -> Optional[Path]:
    # prefer data_wgs.csv
    p = exp_dir / "data_wgs.csv"
    if p.exists():
        return p
    # fallback: any data*.csv
    cands = sorted(exp_dir.glob("data*.csv"))
    return cands[0] if cands else None


def find_model_ckpt(exp_dir: Path, model_type: str) -> Optional[Path]:
    """
    Your folders show: mlp_model.pth sometimes, or mlp_model.pth, or model.pth.
    We'll check a few common names.
    """
    candidates = []
    if model_type == "mlp":
        candidates += [
            exp_dir / "mlp_model.pth",
            exp_dir / "mlp_model.pth",
            exp_dir / "mlp_model.pth",
            exp_dir / "mlp_model.pth",
            exp_dir / "mlp_model.pth",
            exp_dir / "mlp_model.pth",
            exp_dir / "mlp_model.pth",
            exp_dir / "mlp_model.pth",
        ]
        candidates += [exp_dir / "mlp_model.pth", exp_dir / "mlp_model.pth", exp_dir / "mlp_model.pth"]
        candidates += [exp_dir / "mlp_model.pth"]
        candidates += [exp_dir / "mlp_model.pth"]
        candidates += [exp_dir / "mlp_model.pth"]
        candidates += [exp_dir / "mlp_model.pth"]
        candidates += [exp_dir / "mlp_model.pth"]
        # common in your screenshot:
        candidates += [exp_dir / "mlp_model.pth"]
        # also seen in your earlier folder:
        candidates += [exp_dir / "mlp_model.pth"]
        # simple fallback
        candidates += [exp_dir / "mlp_model.pth"]
    else:
        candidates += [
            exp_dir / "model.pth",
            exp_dir / "linear_model.pth",
            exp_dir / "linear_model.pth",
        ]

    # generic fallbacks
    candidates += [
        exp_dir / "model.pth",
        exp_dir / "mlp_model.pth",
        exp_dir / "mlp_model.pth",
        exp_dir / "mlp_model.pth",
        exp_dir / "mlp_model.pth",
        exp_dir / "mlp_model.pth",
    ]

    for c in candidates:
        if c.exists():
            return c

    # last resort: any *.pth
    pths = sorted(exp_dir.glob("*.pth"))
    return pths[0] if pths else None


def infer_model_type(dataset_type: str) -> str:
    return "linear" if dataset_type == "linear_blobs" else "mlp"


def run_attribution_eval(
    exp_dir: Path,
    data_path: Path,
    model_path: Path,
    model_type: str,
    num_classes: int,
    attr_method: str,
    ig_steps: int,
    topk: int,
    sphere_samples: int,
    vis_points: int,
) -> Dict[str, Any]:
    """
    Calls your updated evaluation script that computes boundary metrics + attribution metrics.
    """
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "run_attribution_method.py"),
        "--data_path", str(data_path),
        "--model_path", str(model_path),
        "--model_type", model_type,
        "--num_classes", str(num_classes),
        "--save_dir", str(exp_dir),
        "--vis_dir", str(exp_dir / "vis"),
        "--vis_points", str(vis_points),
        "--sphere_samples", str(sphere_samples),
        "--attr_method", str(attr_method),
        "--ig_steps", str(ig_steps),
        "--topk", str(topk),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        logger.error("FAILED in %s", exp_dir.name)
        logger.error("STDOUT:\n%s", result.stdout)
        logger.error("STDERR:\n%s", result.stderr)
        raise RuntimeError(f"Attribution eval failed for {exp_dir}: {result.stderr[:500]}")

    logger.info("OK: %s", exp_dir.name)
    return {"success": True, "stdout_tail": result.stdout[-500:]}


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Run attribution evaluation across synthetic experiments.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "synthetic_experiments"),
        help="Base directory containing synthetic experiment folders",
    )

    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Run only one dataset folder name (e.g., '15d_moons'). Empty = run all.",
    )

    # Attribution options
    parser.add_argument("--attr_method", type=str, default="ig", choices=["ig", "vg", "sg", "ks", "lime", "none"])
    parser.add_argument("--ig_steps", type=int, default=64)
    parser.add_argument("--topk", type=int, default=10)

    # Evaluation knobs
    parser.add_argument("--sphere_samples", type=int, default=20000)
    parser.add_argument("--vis_points", type=int, default=50)

    parser.add_argument("--skip_existing", action="store_true", help="Skip if report.csv already exists")

    args = parser.parse_args()

    base_dir = Path(args.results_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    datasets = collect_synthetic_experiments_map()

    summary: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "results_dir": str(base_dir),
        "attr_method": args.attr_method,
        "ig_steps": args.ig_steps,
        "topk": args.topk,
        "experiments": [],
    }

    completed = 0
    skipped = 0
    failed = 0

    for name, meta in datasets.items():
        if args.only and name != args.only:
            continue

        exp_dir = base_dir / name
        if not exp_dir.exists():
            logger.warning("Missing folder: %s (skipping)", exp_dir)
            continue

        report_csv = exp_dir / "report.csv"
        if args.skip_existing and report_csv.exists():
            logger.info("Skip existing: %s", name)
            skipped += 1
            continue

        dim = int(meta["dimension"])
        dataset_type = str(meta["dataset_type"])
        n_classes = int(meta["n_classes"])
        model_type = infer_model_type(dataset_type)

        data_path = find_data_csv(exp_dir)
        if data_path is None:
            logger.warning("No data csv found in %s (skipping)", exp_dir)
            skipped += 1
            continue

        model_path = find_model_ckpt(exp_dir, model_type=model_type)
        if model_path is None:
            logger.warning("No model checkpoint found in %s (skipping)", exp_dir)
            skipped += 1
            continue

        try:
            out = run_attribution_eval(
                exp_dir=exp_dir,
                data_path=data_path,
                model_path=model_path,
                model_type=model_type,
                num_classes=n_classes,
                attr_method=args.attr_method,
                ig_steps=args.ig_steps,
                topk=args.topk,
                sphere_samples=args.sphere_samples,
                vis_points=(args.vis_points if dim == 2 else 0),
            )
            summary["experiments"].append(
                {
                    "name": name,
                    "dimension": dim,
                    "dataset_type": dataset_type,
                    "n_classes": n_classes,
                    "model_type": model_type,
                    "data_path": str(data_path),
                    "model_path": str(model_path),
                    "status": "ok",
                    **out,
                }
            )
            completed += 1
        except Exception as e:
            failed += 1
            summary["experiments"].append(
                {
                    "name": name,
                    "dimension": dim,
                    "dataset_type": dataset_type,
                    "n_classes": n_classes,
                    "model_type": model_type,
                    "data_path": str(data_path),
                    "model_path": str(model_path),
                    "status": "failed",
                    "error": str(e),
                }
            )

    summary["completed"] = completed
    summary["skipped"] = skipped
    summary["failed"] = failed

    summary_path = base_dir / "attribution_eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Done. completed=%d skipped=%d failed=%d", completed, skipped, failed)
    logger.info("Summary saved to: %s", summary_path)


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Synthetic Experiments Pipeline

This script generates synthetic datasets with various characteristics,
trains appropriate models, and runs boundary search evaluation.

Dataset Matrix:
- Dimensions: 2D, 8D, 15D
- Types: linear_blobs (2-class), moons (2-class), spirals (2-class), 
         3class (blobs), 10class (blobs)

Model Selection:
- Linear classifier for linear_blobs
- MLP for all curvy/multiclass datasets

Results are saved in organized folders under /results/synthetic_experiments/
with config.json documenting all parameters used.
"""

import os
# Fix OpenMP duplicate library error (MKL + PyTorch both load libiomp5md.dll)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data.dataset_utils import (
    generate_linearly_separable_data,
    generate_moons_data,
    generate_spirals_data,
    generate_multiclass_blobs_data,
    generate_high_dim_curvy_data,
    generate_concentric_hyperspheres_data,
    save_dataset_to_csv,
)
from utils.models.train_mlp_model import train_mlp_model_memory
from utils.models.train_linear_model import train_model_memory as train_linear_model_memory


# Parameter configurations by dimension
DIMENSION_PARAMS = {
    2: {
        "gs_r_max": 5.0,
        "gs_samples": 512,
        "gs_dirs": 128,
        "max_iters": 50,
        "step_size": 0.01,
        "training_epochs": 50,
    },
    8: {
        "gs_r_max": 10.0,
        "gs_samples": 1024,
        "gs_dirs": 256,
        "max_iters": 100,
        "step_size": 0.005,
        "training_epochs": 100,
    },
    15: {
        "gs_r_max": 15.0,
        "gs_samples": 2048,
        "gs_dirs": 512,
        "max_iters": 150,
        "step_size": 0.003,
        "training_epochs": 150,
    },
}

# Dataset types and their generators
DATASET_TYPES = [
    "linear_blobs",  # 2-class linearly separable
    "moons",         # 2-class curvy (2D only, hyperspheres for high-D)
    "spirals",       # 2-class curvy (2D only, sectors for high-D)
    "3class",        # 3-class blobs
    "10class",       # 10-class blobs
]

DIMENSIONS = [2, 8, 15]


def generate_dataset(
    dataset_type: str,
    n_features: int,
    n_samples_per_class: int = 200,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate a dataset based on type and dimensionality.
    
    Returns:
        X: Feature matrix
        y: Labels
        metadata: Dictionary with dataset parameters
    """
    metadata = {
        "dataset_type": dataset_type,
        "n_features": n_features,
        "n_samples_per_class": n_samples_per_class,
        "random_state": random_state,
    }
    
    if dataset_type == "linear_blobs":
        X, y = generate_linearly_separable_data(
            n_samples_per_class=n_samples_per_class,
            n_features=n_features,
            random_state=random_state,
        )
        metadata["n_classes"] = 2
        metadata["generator"] = "generate_linearly_separable_data"
        
    elif dataset_type == "moons":
        if n_features == 2:
            X, y = generate_moons_data(
                n_samples_per_class=n_samples_per_class,
                noise=0.1,
                random_state=random_state,
            )
            metadata["generator"] = "generate_moons_data"
            metadata["noise"] = 0.1
        else:
            # Use concentric hyperspheres for high-D "moons-like" boundary
            X, y = generate_high_dim_curvy_data(
                n_samples_per_class=n_samples_per_class,
                n_features=n_features,
                n_classes=2,
                pattern="concentric",
                noise=0.1,
                random_state=random_state,
            )
            metadata["generator"] = "generate_high_dim_curvy_data"
            metadata["pattern"] = "concentric"
            metadata["noise"] = 0.1
        metadata["n_classes"] = 2
        
    elif dataset_type == "spirals":
        if n_features == 2:
            X, y = generate_spirals_data(
                n_samples_per_class=n_samples_per_class,
                noise=0.2,
                n_rotations=2.0,
                random_state=random_state,
            )
            metadata["generator"] = "generate_spirals_data"
            metadata["noise"] = 0.2
            metadata["n_rotations"] = 2.0
        else:
            # Use sector pattern for high-D "spiral-like" boundary
            X, y = generate_high_dim_curvy_data(
                n_samples_per_class=n_samples_per_class,
                n_features=n_features,
                n_classes=2,
                pattern="sectors",
                noise=0.1,
                random_state=random_state,
            )
            metadata["generator"] = "generate_high_dim_curvy_data"
            metadata["pattern"] = "sectors"
            metadata["noise"] = 0.1
        metadata["n_classes"] = 2
        
    elif dataset_type == "3class":
        X, y = generate_multiclass_blobs_data(
            n_samples_per_class=n_samples_per_class,
            n_features=n_features,
            n_classes=3,
            cluster_std=1.0,
            random_state=random_state,
        )
        metadata["n_classes"] = 3
        metadata["generator"] = "generate_multiclass_blobs_data"
        metadata["cluster_std"] = 1.0
        
    elif dataset_type == "10class":
        X, y = generate_multiclass_blobs_data(
            n_samples_per_class=n_samples_per_class,
            n_features=n_features,
            n_classes=10,
            cluster_std=1.0,
            random_state=random_state,
        )
        metadata["n_classes"] = 10
        metadata["generator"] = "generate_multiclass_blobs_data"
        metadata["cluster_std"] = 1.0
        
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    metadata["total_samples"] = len(X)
    return X, y, metadata


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    save_path: str,
    epochs: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train a model on the given data.
    
    Returns:
        metadata: Dictionary with training parameters
    """
    metadata = {
        "model_type": model_type,
        "epochs": epochs,
        "seed": seed,
        "save_path": str(save_path),
    }
    
    if model_type == "mlp":
        model = train_mlp_model_memory(
            X_train=X,
            y_train=y.astype(np.int64),
            epochs=epochs,
            seed=seed,
            save_path=save_path,
        )
    elif model_type == "linear":
        model = train_linear_model_memory(
            X_train=X,
            y_train=y.astype(np.int64),
            epochs=epochs,
            seed=seed,
            save_path=save_path,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return metadata


def run_boundary_search_eval(
    data_path: str,
    model_path: str,
    model_type: str,
    save_dir: str,
    n_features: int,
    num_classes: int,
) -> Dict[str, Any]:
    """
    Run crawler boundary search evaluation script.
    
    Returns:
        metadata: Dictionary with evaluation parameters
    """
    params = DIMENSION_PARAMS.get(n_features, DIMENSION_PARAMS[2])
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "experiments" / "run_crawler_boundary_search_eval.py"),
        "--data_path", str(data_path),
        "--model_path", str(model_path),
        "--model_type", model_type,
        "--save_dir", str(save_dir),
        "--num_classes", str(num_classes),
        "--gs_r_max", str(params["gs_r_max"]),
        "--gs_dirs", str(params["gs_dirs"]),
        "--fgsm_max_iters", str(params["max_iters"]),
        "--fgsm_step_size", str(params["step_size"]),
        "--vis_dir", str(Path(save_dir) / "vis"),
        "--vis_points", "50" if n_features == 2 else "0",
    ]
    
    metadata = {
        "method": "crawler",
        "gs_r_max": params["gs_r_max"],
        "gs_dirs": params["gs_dirs"],
        "fgsm_max_iters": params["max_iters"],
        "fgsm_step_size": params["step_size"],
    }
    
    print(f"Running: {' '.join(cmd)}")
    
    # Pass PYTHONPATH so subprocess can find project modules
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode != 0:
        print(f"ERROR: Boundary search failed!")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Boundary search evaluation failed: {result.stderr[:500]}")
    else:
        metadata["success"] = True
        print(result.stdout)
    
    return metadata


def run_experiment(
    dataset_type: str,
    n_features: int,
    base_results_dir: Path,
    n_samples_per_class: int = 200,
    random_state: int = 42,
    skip_eval: bool = False,
) -> Dict[str, Any]:
    """
    Run a single experiment: generate data, train model, run evaluation.
    """
    experiment_name = f"{n_features}d_{dataset_type}"
    experiment_dir = base_results_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name}")
    print(f"{'='*60}")
    
    config = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "n_features": n_features,
        "dataset_type": dataset_type,
    }
    
    # Step 1: Generate dataset
    print(f"\n[1/3] Generating dataset...")
    X, y, dataset_metadata = generate_dataset(
        dataset_type=dataset_type,
        n_features=n_features,
        n_samples_per_class=n_samples_per_class,
        random_state=random_state,
    )
    config["dataset"] = dataset_metadata
    
    # Save dataset
    data_path = experiment_dir / "data.csv"
    save_dataset_to_csv(X, y, out_dir=experiment_dir, prefix="data", timestamp="")
    # Rename to consistent name
    generated_files = list(experiment_dir.glob("data_*.csv"))
    if generated_files:
        generated_files[0].rename(data_path)
    print(f"  Dataset saved to: {data_path}")
    print(f"  Shape: X={X.shape}, y={y.shape}, classes={len(np.unique(y))}")
    
    # Step 2: Train model
    print(f"\n[2/3] Training model...")
    model_type = "linear" if dataset_type == "linear_blobs" else "mlp"
    model_path = experiment_dir / f"{model_type}_model.pth"
    
    training_params = DIMENSION_PARAMS.get(n_features, DIMENSION_PARAMS[2])
    
    training_metadata = train_model(
        X=X,
        y=y,
        model_type=model_type,
        save_path=str(model_path),
        epochs=training_params["training_epochs"],
        seed=random_state,
    )
    config["training"] = training_metadata
    print(f"  Model saved to: {model_path}")
    
    # Step 3: Run boundary search evaluation
    if not skip_eval:
        print(f"\n[3/3] Running boundary search evaluation...")
        eval_metadata = run_boundary_search_eval(
            data_path=str(data_path),
            model_path=str(model_path),
            model_type=model_type,
            save_dir=str(experiment_dir),
            n_features=n_features,
            num_classes=dataset_metadata["n_classes"],
        )
        config["evaluation"] = eval_metadata
    else:
        print(f"\n[3/3] Skipping boundary search evaluation (--skip-eval)")
        config["evaluation"] = {"skipped": True}
    
    # Save config
    config_path = experiment_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to: {config_path}")
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Run synthetic experiments pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python run_synthetic_experiments.py
  
  # Run only 2D experiments
  python run_synthetic_experiments.py --dimensions 2
  
  # Run only moons and spirals
  python run_synthetic_experiments.py --types moons spirals
  
  # Dry run (only generate data and train models, skip evaluation)
  python run_synthetic_experiments.py --skip-eval
        """
    )
    
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=DIMENSIONS,
        help=f"Dimensions to run experiments for (default: {DIMENSIONS})"
    )
    parser.add_argument(
        "--types",
        type=str,
        nargs="+",
        default=DATASET_TYPES,
        choices=DATASET_TYPES,
        help=f"Dataset types to run (default: {DATASET_TYPES})"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "synthetic_experiments"),
        help="Base directory for results"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Number of samples per class (default: 200)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip boundary search evaluation (useful for testing data generation)"
    )
    
    args = parser.parse_args()
    
    base_results_dir = Path(args.results_dir)
    base_results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Synthetic Experiments Pipeline")
    print("=" * 60)
    print(f"Dimensions: {args.dimensions}")
    print(f"Dataset types: {args.types}")
    print(f"Results directory: {base_results_dir}")
    print(f"Samples per class: {args.n_samples}")
    print(f"Random seed: {args.seed}")
    print(f"Boundary search method: crawler")
    print(f"Skip evaluation: {args.skip_eval}")
    
    # Count total experiments
    total_experiments = len(args.dimensions) * len(args.types)
    print(f"\nTotal experiments to run: {total_experiments}")
    
    # Run experiments
    all_configs = []
    completed = 0
    failed = 0
    
    for dim in args.dimensions:
        for dtype in args.types:
            try:
                config = run_experiment(
                    dataset_type=dtype,
                    n_features=dim,
                    base_results_dir=base_results_dir,
                    n_samples_per_class=args.n_samples,
                    random_state=args.seed,
                    skip_eval=args.skip_eval,
                )
                all_configs.append(config)
                completed += 1
            except RuntimeError:
                raise  # Re-raise to stop pipeline on boundary search failures
            except Exception as e:
                print(f"\nERROR in {dim}d_{dtype}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": total_experiments,
        "completed": completed,
        "failed": failed,
        "experiments": all_configs,
    }
    
    summary_path = base_results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Completed: {completed}/{total_experiments}")
    print(f"Failed: {failed}/{total_experiments}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

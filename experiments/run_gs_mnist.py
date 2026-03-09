from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch

from boundary_search.growing_spheres_oracle import GrowingSpheresOracle
from utils.data.dataset_utils import get_mnist_dataloaders
from utils.entities.small_mnist_cnn import SmallMNISTCNN

logging.basicConfig(
    level=logging.INFO,  # switch to DEBUG when you need details
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

@torch.no_grad()
def collect_correct_samples(model, loader, device, max_samples: int):
    model.eval()
    samples = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)

        for i in range(x.size(0)):
            if int(preds[i]) == int(y[i]):
                samples.append((x[i].detach().cpu(), int(y[i])))
                if len(samples) >= max_samples:
                    return samples
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data") #/Users/nguyennhatmai/Documents/study/UBonn/WiSe2526/LabDMAI/local-boundary-attribution/data
    parser.add_argument("--model_dir", type=str, required=True) #/Users/nguyennhatmai/Documents/study/UBonn/WiSe2526/LabDMAI/local-boundary-attribution/models/checkpoints/mnist
    parser.add_argument("--output_csv", type=str, default="./results/mnist/growing_sphere_results.csv") # /Users/nguyennhatmai/Documents/study/UBonn/WiSe2526/LabDMAI/local-boundary-attribution/data/MNIST
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--digit_a", type=int, default=None)
    parser.add_argument("--digit_b", type=int, default=None)

    parser.add_argument("--n_directions", type=int, default=15*4096)
    parser.add_argument("--r_init", type=float, default=0.01)
    parser.add_argument("--r_step", type=float, default=1)
    parser.add_argument("--r_max", type=float, default=20.0)
    parser.add_argument("--boundary_bisect_steps", type=int, default=30)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    with open(model_dir / "train_config.json", "r", encoding="utf-8") as f:
        train_cfg = json.load(f)

    binary_digits = None
    if args.digit_a is not None and args.digit_b is not None:
        binary_digits = (args.digit_a, args.digit_b)
    elif train_cfg.get("binary_digits") is not None:
        binary_digits = tuple(train_cfg["binary_digits"])

    _, test_loader, num_classes = get_mnist_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        binary_digits=binary_digits,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SmallMNISTCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))
    model.eval()

    oracle = GrowingSpheresOracle(
        model=model,
        n_directions=args.n_directions,
        r_init=args.r_init,
        r_step=args.r_step,
        r_max=args.r_max,
        boundary_bisect_steps=args.boundary_bisect_steps,
        clamp=(0.0, 1.0),
        device=device,
        seed=train_cfg.get("seed", 0),
    )

    samples = collect_correct_samples(model, test_loader, device, args.max_samples)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_idx",
        "num_features",
        "x0",
        "class",
        "gs_success",
        "found_radius",
        "x_enemy",
        "x_boundary",
        "dist_x0_enemy",
        "dist_x0_boundary",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        logger.info("Evaluating %d points", len(samples))
        for sample_idx, (x0_t, y) in enumerate(samples):
            x0 = x0_t.numpy().reshape(-1).astype(np.float32)

            result = oracle.find_boundary(x0, y=y)

            dist_x0_enemy = float(np.linalg.norm(result.x_start - result.x_enemy))
            dist_x0_boundary = float(np.linalg.norm(result.x_start - result.x_boundary))

            row = {
                "sample_idx": sample_idx,
                "num_features": int(x0.shape[0]),
                "x0": result.x_start.tolist(),
                "class": int(y),
                "gs_success": bool(result.success),
                "found_radius": float(result.radius_found) if result.success else "",
                "x_enemy": result.x_enemy.tolist() if result.success else "",
                "x_boundary": result.x_boundary.tolist() if result.success else "",
                "dist_x0_enemy": dist_x0_enemy if result.success else "",
                "dist_x0_boundary": dist_x0_boundary if result.success else "",
            }
            writer.writerow(row)
            logging.info(f"Processed sample {sample_idx + 1}/{len(samples)} | gs_success={result.success}")

    logging.info(f"Saved Growing Spheres results to: {output_csv}")


if __name__ == "__main__":
    main()
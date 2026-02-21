import logging
import argparse

import numpy as np
import pandas as pd
import torch

from evaluation import GrowingSpheresOracle
from utils import load_dataset_from_csv, load_model
from utils.common.file_utils import find_data_csv_folders, to_row

logging.basicConfig(
    level=logging.INFO,  # switch to DEBUG when you need details
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # IO
    parser.add_argument("--root_directory", type=str, required=True, default="/Users/nguyennhatmai/Documents/study/UBonn/WiSe2526/LabDMAI/local-boundary-attribution/results")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "linear"])

    # GS oracle params
    parser.add_argument("--gs_dirs", type=int, default=1024)
    parser.add_argument("--gs_r_init", type=float, default=0.2)
    parser.add_argument("--gs_r_step", type=float, default=0.05)
    parser.add_argument("--gs_r_max", type=float, default=16.0)
    parser.add_argument("--gs_bisect_steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    root_directory = args.root_directory
    folders = find_data_csv_folders(root_directory)

    all_rows = []

    for folder in folders:
        logging.info(folder)
        data_path = folder + '/data.csv'
        model_path = folder + '/mlp_model.pth'

        X, y = load_dataset_from_csv(data_path)  # X: (N,d), y: (N,)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        d = X.shape[1]
        num_classes = int(len(np.unique(y)))

        model = load_model(model_path, model_type=args.model_type, input_dim=d, num_classes=num_classes)
        # model = load_model(model_path, model_type=args.model_type)
        model = model.to(device)
        model.eval()

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

        # n = min(len(X), args.max_points)
        n = len(X)
        logger.info("Evaluating %d points", n)
        for i in range(n):
            x_i = np.asarray(X[i], dtype=np.float32)
            y_i = int(y[i])

            gs_res = gs_oracle.find_boundary(x_i, y=y_i)

            row = to_row(
                x0=x_i,
                y_true=y_i,
                gs_res=gs_res,
            )
            all_rows.append(row)

        out_path = folder + "/data_wgs.csv"
        out_df = pd.DataFrame(all_rows)
        out_df.to_csv(out_path, index=False)
        logger.info("Wrote aggregated GS results to: %s (rows=%d)", out_path, len(out_df))
